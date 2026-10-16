from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import numpy as np

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None, return_attn=False):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        self.attn_map = attn.detach()

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        if return_attn:
            return self.to_out(out), attn
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        # 1. 原始 self-attention
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None, use_hier=True):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        cls = HierarchicalBasicTransformerBlock if use_hier else BasicTransformerBlock
        self.transformer_blocks = nn.ModuleList(
            [cls(inner_dim, n_heads, d_head, dropout=dropout, 
                context_dim=context_dim) for _ in range(depth)]
)

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

        # ================= 方案B新增：旁路特征投影与缓存 =================
        self.hier_proj = zero_module(nn.Conv2d(512, inner_dim, kernel_size=1))
        self.hier_feat_map = None  # 由外部注入
        # ============================================================

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # 注入旁路特征到每个 block
        if self.hier_feat_map is not None:
            key = (h, w)
            if key in self.hier_feat_map:
                hier_feat = self.hier_proj(self.hier_feat_map[key])
                hier_flat = rearrange(hier_feat, 'b c h w -> b (h w) c')
                for block in self.transformer_blocks:
                    if isinstance(block, HierarchicalBasicTransformerBlock):
                        block.hier_feat = hier_flat
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in


def replace_blocks_with_hierarchical(model, max_depth=3, device='cuda'):
    """替换模型中的BasicTransformerBlock为Hierarchical版本"""
    for name, module in model.named_children():
        if isinstance(module, (BasicTransformerBlock, HierarchicalBasicTransformerBlock)):
            # 创建新的层级block
            dim      = module.attn1.to_q.in_features
            n_heads  = module.attn1.heads
            d_head   = dim // n_heads
            ctx_dim  = module.attn2.to_k.in_features
            hierarchical_block = HierarchicalBasicTransformerBlock(
                dim=dim,
                n_heads=n_heads,
                d_head=d_head,
                context_dim=ctx_dim,
                gated_attention=True
            )
            
            # 复制权重
            hierarchical_block.load_state_dict(module.state_dict(), strict=False)
            hierarchical_block = hierarchical_block.to(device)
            
            # 替换当前模块
            setattr(model, name, hierarchical_block)
        else:
            # 递归替换子模块
            replace_blocks_with_hierarchical(module, max_depth, device)

# 在采样前注入depth_map
def inject_depth_map(model, depth_map, ancestry_map, parent_map, timestep=None, max_t=None):
    for module in model.modules():
        if isinstance(module, HierarchicalBasicTransformerBlock):
            module.depth_map   = depth_map
            module.ancestry_map = ancestry_map
            module.parent_map   = parent_map      # dict[str, str]
            module.current_t     = timestep
            module.max_t        = max_t  # 🔥 新增

def inject_hier_feats(model, hier_feats_dict):
    """
    将旁路网络输出的多尺度特征注入到所有 SpatialTransformer
    hier_feats_dict: {(H, W): Tensor}
    """
    for module in model.modules():
        if isinstance(module, SpatialTransformer):
            module.hier_feat_map = hier_feats_dict


class HierarchicalBasicTransformerBlock(BasicTransformerBlock):
    gate_records = []
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None,
                 gated_ff=True, checkpoint=True, gated_attention=True):
        super().__init__(dim, n_heads, d_head, dropout, context_dim, gated_ff, checkpoint)

        self.gated_attention = gated_attention
        if self.gated_attention:
            self.gate_weight = nn.Parameter(torch.tensor(0.0))

        # 外部注入用
        self.depth_map    = None
        self.ancestry_map = None
        self.parent_map   = None
        self.hier_feat = None  # [B, L, C] 旁路特征（已投影到当前维度）

        # ================= 新增：壳层摘要融合层 =================
        # 将"旁路壳层特征"与"当前特征 max pooling"融合
        self.shell_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        # 零初始化：训练初期旁路不干扰，逐渐接管
        nn.init.zeros_(self.shell_fusion[2].weight)
        nn.init.zeros_(self.shell_fusion[2].bias)
        # =====================================================

    def forward(self, x, context=None):
        if self.checkpoint:
            import torch.utils.checkpoint as cp
            return cp.checkpoint(self._forward, x, context, use_reentrant=False)
        else:
            return self._forward(x, context)

    def _forward(self, x, context=None):
        B, L, C = x.shape
        H = int(np.sqrt(L))
        
        # === 共享投影 ===
        x_norm = self.norm1(x)
        q = self.attn1.to_q(x_norm)
        k = self.attn1.to_k(x_norm)
        v = self.attn1.to_v(x_norm)
        
        # === 辅助函数：统一attention计算 ===
        def compute_attn(q, k, v, mask=None):
            qh, kh, vh = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=self.attn1.heads), (q, k, v))
            sim = torch.einsum('b i d, b j d -> b i j', qh, kh) * self.attn1.scale
            if mask is not None:
                h = self.attn1.heads
                if mask.dim() == 3:
                    mask = mask.unsqueeze(1)
                mask = repeat(mask, 'b 1 i j -> (b h) i j', h=h)
                sim = sim.masked_fill(mask, -torch.inf)
            attn = sim.softmax(dim=-1)
            out = torch.einsum('b i j, b j d -> b i d', attn, vh)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=self.attn1.heads)
            return self.attn1.to_out(out), attn
        
        # === 全局分支（无mask）===
        out_global, attn_global = compute_attn(q, k, v)
        self.global_attn = attn_global.mean(dim=0).detach()


        if self.gated_attention and self.depth_map is not None:
            B, L, C = x.shape
            H = int(np.sqrt(L))
            
            # 1. 空间映射
            depth_flat = F.interpolate(self.depth_map.float(), size=(H,H), mode='nearest').long().view(B, L)
            ancestry_flat = F.interpolate(self.ancestry_map.float(), size=(H,H), mode='nearest').long().view(B, L)
            
            # 2. 构建 parent_of 和 children_of，并生成 k_hier / v_hier
            k_hier = k.clone()
            v_hier = v.clone()

            shell_map_flat = torch.full_like(ancestry_flat, -1)
            
            for b in range(B):
                # 取当前样本的 parent_map {child_id: parent_id}
                pmap = self.parent_map[b] if isinstance(self.parent_map, list) else self.parent_map
                if not isinstance(pmap, dict):
                    continue
                
                # 统一为标量int
                parent_of = {}
                for cid, pid in pmap.items():
                    cid = int(cid.item() if isinstance(cid, torch.Tensor) else cid)
                    pid = int(pid.item() if isinstance(pid, torch.Tensor) else pid)
                    parent_of[cid] = pid
                
                # 反转：parent_id -> list of child_ids（用于减壳层）
                children_of = {}
                for cid, pid in parent_of.items():
                    children_of.setdefault(pid, []).append(cid)
                
                unique_ids = torch.unique(ancestry_flat[b])
                
                for uid in unique_ids:
                    uid_int = int(uid.item())
                    if uid_int <= 0:
                        continue
                    
                    # 跳过根节点
                    uid_depths = depth_flat[b, ancestry_flat[b] == uid_int].unique()
                    if uid_depths.numel() == 0 or int(uid_depths[0].item()) <= 1:
                        continue
                    
                    # 找直接父
                    parent_id = parent_of.get(uid_int, -1)
                    if parent_id <= 0:
                        continue
                    
                    # 父区域
                    parent_region = (ancestry_flat[b] == parent_id)
                    
                    # 🔥 真壳层：父区域 减去 该父的所有子组件（含兄弟和自己）
                    child_ids = children_of.get(parent_id, [])
                    child_region = torch.zeros_like(parent_region)
                    for cid in child_ids:
                        child_region |= (ancestry_flat[b] == cid)
                    
                    shell_mask = parent_region & (~child_region)
                    if shell_mask.sum() == 0:
                        continue
                    
                    # === 🔥 关键修改：壳层摘要生成 ===
                    # 方式：旁路特征 + 当前特征融合（替代纯 max pooling）
                    
                    # (a) 当前特征的 max pooling（保底）
                    z_k_local = k[b, shell_mask].max(dim=0, keepdim=True)[0]  # [1, C]
                    z_v_local = v[b, shell_mask].max(dim=0, keepdim=True)[0]
                    
                    # (b) 旁路特征在壳层区域的聚合（如果可用）
                    if self.hier_feat is not None:
                        hier_shell = self.hier_feat[b, shell_mask]  # [N_shell, C]
                        z_k_hier = hier_shell.mean(dim=0, keepdim=True)  # [1, C]
                        z_v_hier = hier_shell.mean(dim=0, keepdim=True)
                        
                        # (c) 可学习融合：旁路特征 + 当前特征
                        z_k_fused = self.shell_fusion(torch.cat([z_k_local, z_k_hier], dim=-1))
                        z_v_fused = self.shell_fusion(torch.cat([z_v_local, z_v_hier], dim=-1))
                    else:
                        # 无旁路时 fallback 到原始 max pooling
                        z_k_fused = z_k_local
                        z_v_fused = z_v_local
                    
                    # 填入壳层
                    k_hier[b] = torch.where(shell_mask.unsqueeze(-1), z_k_fused, k_hier[b])
                    v_hier[b] = torch.where(shell_mask.unsqueeze(-1), z_v_fused, v_hier[b])
                    if shell_mask.sum() > 0:
                        shell_map_flat[b, shell_mask] = parent_id
            
            # 3. 构建 parent_flat [B, L]：每个位置的直接父ID
            parent_flat = torch.full_like(ancestry_flat, -1)
            for b in range(B):
                pmap = self.parent_map[b] if isinstance(self.parent_map, list) else self.parent_map
                for pos in range(L):
                    cid = int(ancestry_flat[b, pos].item())
                    if cid > 0:
                        pid = pmap.get(cid, -1)
                        parent_flat[b, pos] = int(pid.item() if isinstance(pid, torch.Tensor) else pid)
            
            # 5. 🔥 精确Mask：子组件只能查父壳层（ancestry_k == parent_q 且 depth_k == parent_depth）
            parent_q = parent_flat.unsqueeze(2).expand(-1, -1, L)          # [B,L,L]
            shell_k = shell_map_flat.unsqueeze(1).expand(-1, L, -1)
            
            # key必须是父组件本身（深度=父深度），不能是子组件（深度更大）
            is_parent_shell = (shell_k == parent_q) & (parent_q > -1)
            
            # 严禁看自己
            is_self = (ancestry_flat.unsqueeze(2) == ancestry_flat.unsqueeze(1))
            
            valid_mask = is_parent_shell & (~is_self)
            # 排除背景
            is_bg = (ancestry_flat <= 0)
            valid_mask = valid_mask & ~is_bg.unsqueeze(2) & ~is_bg.unsqueeze(1)
            
            # 保底
            first_valid = valid_mask.max(dim=-1, keepdim=True)[1]
            valid_mask = valid_mask.scatter(-1, first_valid, True)
            attn_mask = ~valid_mask.unsqueeze(1)  # [B,1,L,L]  True=屏蔽
            
            # 6. 层级attention
            out_hier, attn_hier = compute_attn(q, k_hier, v_hier, attn_mask)
            self.parent_attn = attn_hier.mean(dim=0).detach()
            
            # 7.1 计算每个组件的风险分数 r_c，并映射到每个 token
            a_ref = getattr(self, 'a_ref', 0.0066)   # 建议先从训练集统计 P25({a_c}) 再设置
            rho = getattr(self, 'rho', 0.25)       # 消融实验取 {0, 0.25, 0.5}
            
            r_flat = torch.zeros(B, L, device=x.device, dtype=x.dtype)
            
            for b in range(B):
                # 当前页面最大深度
                d_max = int(depth_flat[b].max().item())
                if d_max <= 2:
                    d_max = 2
                
                # 统计当前页面各组件 latent 面积占比 a_c
                unique_ids = torch.unique(ancestry_flat[b])
                areas = {}
                for uid in unique_ids:
                    uid_int = int(uid.item())
                    if uid_int <= 0:
                        continue
                    mask = (ancestry_flat[b] == uid_int)
                    a_c = mask.sum().item() / L
                    areas[uid_int] = a_c
                
                # 逐组件计算风险并赋给对应 token
                for uid_int, a_c in areas.items():
                    # 深度风险 d_hat
                    d_vals = depth_flat[b, ancestry_flat[b] == uid_int].unique()
                    if d_vals.numel() == 0:
                        continue
                    d_c = int(d_vals[0].item())
                    
                    if d_max > 2:
                        d_hat = max(0.0, (d_c - 2.0) / (d_max - 2.0))
                    else:
                        d_hat = 0.0
                    
                    # 面积风险 a_hat
                    a_hat = 1.0 - min(1.0, a_c / a_ref)
                    
                    # 综合风险 r_c = 0.5 * d_hat + 0.5 * a_hat
                    r_c = 0.5 * d_hat + 0.5 * a_hat
                    
                    # 赋给该组件所有 token（背景保持 0）
                    comp_mask = (ancestry_flat[b] == uid_int)
                    r_flat[b, comp_mask] = r_c
            
            # 7.2 基础 gate = 0.3 + 风险自适应 ρ
            gate = torch.sigmoid(self.gate_weight)  # scalar
            lambda_s0 = 1-gate
            lambda_p0 = gate

            lambda_s = lambda_s0 * (1.0 + rho * r_flat)
            lambda_p = lambda_p0 * (1.0 - rho * r_flat)
            
            # 7.3 自适应门控融合：H'_c = H_c + λ_s(c) * A_c^self + λ_p(c) * A_{c←p}
            # out_global 对应 A_c^self（全局/自身结构分支），out_hier 对应 A_{c←p}（父继承分支）
            lambda_s = lambda_s.unsqueeze(-1)  # [B, L, 1]
            lambda_p = lambda_p.unsqueeze(-1)  # [B, L, 1]
            
            x = x + lambda_s * out_global + lambda_p * out_hier

        else:
            x = x + out_global
            self.parent_attn = None

        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x