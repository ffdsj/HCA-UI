import os
import json
import numpy as np
import glob
from torchvision import transforms
from torch.utils.data import Dataset
from typing import Dict, List, Tuple
import torch
from PIL import Image

# 从之前的文件导入解析函数
from json_parse import parse_json_tree


def create_maps_from_ui_tree(nodes, img_wh, latent_size):
    """
    根据UI树节点生成层级掩码图
    返回: (depth_map, ancestry_map, parent_dict)
    """
    depth_map = np.zeros((latent_size, latent_size), np.float32)
    ancest_map = np.full((latent_size, latent_size), -1, np.int32)  # 背景用 -1
    parent_dict = {}  # child_id -> parent_id
    
    node_id_counter = 1
    node_id_map = {}  # clean_id -> assigned_id

    # 按深度升序（父在前，子在后）——子后填覆盖父
    sorted_nodes = sorted(nodes, key=lambda x: x['ancestry_id'].count('-'))
    
    # 先构建所有ID映射（确保父子ID都存在）
    for node in nodes:
        clean_id = node['ancestry_id'].split('|')[0]
        if clean_id not in node_id_map:
            node_id_map[clean_id] = node_id_counter
            node_id_counter += 1

    for node in sorted_nodes:
        if 'ancestry_id' not in node or 'bounds' not in node:
            continue
            
        depth = node['ancestry_id'].count('-')
        x1, y1, x2, y2 = node['bounds']
        
        # 坐标缩放到latent空间（64x64）
        x1_l = int(x1 * latent_size / img_wh[0])
        y1_l = int(y1 * latent_size / img_wh[1])
        x2_l = int(x2 * latent_size / img_wh[0])
        y2_l = int(y2 * latent_size / img_wh[1])
        
        
        # ========== 修改后 ==========
        # 确保至少保留一个像素（防止小组件在latent空间消失）
        if x1_l >= x2_l:
            x2_l = x1_l + 1
        if y1_l >= y2_l:
            y2_l = y1_l + 1
        # 再clip到边界
        x1_l = min(max(x1_l, 0), latent_size - 1)
        y1_l = min(max(y1_l, 0), latent_size - 1)
        x2_l = min(max(x2_l, x1_l + 1), latent_size)
        y2_l = min(max(y2_l, y1_l + 1), latent_size)

        # 1. 深度图
        depth_map[y1_l:y2_l, x1_l:x2_l] = depth
        
        # 2. 自增 node_id（去除可能的|后缀）
        clean_id = node['ancestry_id'].split('|')[0]
        if clean_id not in node_id_map:
            node_id_map[clean_id] = node_id_counter
            node_id_counter += 1
        node_id = node_id_map[clean_id]
        
        ancest_map[y1_l:y2_l, x1_l:x2_l] = node_id
        
        # 3. 父节点映射（解析"1-1-1" -> "1-1"）
        parent_clean = '-'.join(clean_id.split('-')[:-1])
        if parent_clean:
            if parent_clean not in node_id_map:
                node_id_map[parent_clean] = node_id_counter
                node_id_counter += 1
            parent_id = node_id_map[parent_clean]
        else:
            parent_id = -1  # 根节点
            
        parent_dict[node_id] = parent_id
    
    # 确保父ID在map中存在，被子查询时能命中
    for node in sorted_nodes:
        clean_id = node['ancestry_id'].split('|')[0]
        node_id = node_id_map[clean_id]
        
        # 如果该节点是父节点（被其他节点引用）
        if node_id in parent_dict.values():
            x1, y1, x2, y2 = node['bounds']
            cx_l = min(max(int((x1+x2)/2 * latent_size / img_wh[0]), 0), latent_size-1)
            cy_l = min(max(int((y1+y2)/2 * latent_size / img_wh[1]), 0), latent_size-1)
            ancest_map[cy_l, cx_l] = node_id  # 强制保留父中心点

    return (
        torch.from_numpy(depth_map)[None],    # [1, 1, 64, 64]
        torch.from_numpy(ancest_map)[None],   # [1, 1, 64, 64]
        parent_dict
    )


class UITreeDataset(Dataset):
    """
    统一加载UI图像、文本描述和JSON层级结构
    返回LDM训练所需的完整数据包（含层级掩码）
    """
    def __init__(
        self, 
        image_dir: str, 
        json_dir: str, 
        text_dir: str,  # 如果为None，使用json中的text字段或空字符串
        size: int = 256,
        img_wh: Tuple[int, int] = None,  # 原始UI分辨率
        latent_size: int = 64,  # SD latent空间大小（256->64, 512->64）
        augment: bool = False
    ):
        super().__init__()
        self.image_dir = image_dir
        self.json_dir = json_dir
        self.text_dir = text_dir
        self.size = size
        self.img_wh = img_wh
        self.latent_size = latent_size
        
        # 获取文件列表（按数字ID排序）
        self.samples = self._scan_files()
        
        # 图像预处理
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
    
    def _scan_files(self) -> List[Dict]:
        """扫描并匹配图像-JSON-文本文件"""
        samples = []
        
        # 获取所有图像文件
        img_exts = ['*.png', '*.jpg', '*.jpeg']
        image_files = []
        for ext in img_exts:
            image_files.extend(glob.glob(os.path.join(self.image_dir, ext)))
        
        for img_path in image_files:
            basename = os.path.splitext(os.path.basename(img_path))[0]
            
            # 查找对应JSON
            json_path = os.path.join(self.json_dir, f"{basename}.json")
            if not os.path.exists(json_path):
                continue
            
            # 查找对应文本（可选）
            txt_path = os.path.join(self.text_dir, f"{basename}.txt")
            if not os.path.exists(txt_path):
                raise FileNotFoundError(f"缺少对应文本文件: {txt_path}")
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
            
            samples.append({
                'id': basename,
                'image': img_path,
                'json': json_path,
                'text': text
            })
        
        # 按ID排序
        samples.sort(key=lambda x: int(x['id']) if x['id'].isdigit() else x['id'])
        print(f"[UITreeDataset] 找到 {len(samples)} 个有效样本")
        return samples
    
    def __len__(self):
        return len(self.samples)

    def _get_json_resolution(self, json_data: dict) -> tuple:
        """
        从JSON中提取根节点分辨率，兼容不同格式
        返回: (width, height)
        """
        # 适配100.json（嵌套在activity.root）
        if 'activity' in json_data and 'root' in json_data['activity']:
            root = json_data['activity']['root']
        else:
            root = json_data  # 10.json格式
        
        # 获取根节点bounds [x1,y1,x2,y2]
        bounds = root.get('bounds', [0, 0, 1440, 2560])
        w, h = bounds[2], bounds[3]
        
        # 防御：如果根节点bounds异常（如[0,0,0,0]），找第一个有效子节点
        if w <= 0 or h <= 0:
            def find_valid_bounds(node):
                b = node.get('bounds', [0,0,0,0])
                if b[2] > b[0] and b[3] > b[1]:
                    return (b[2]-b[0], b[3]-b[1])  # 返回wh而非x2y2
                for child in node.get('children', []):
                    res = find_valid_bounds(child)
                    if res:
                        return res
                return None
            
            fallback = find_valid_bounds(root)
            if fallback:
                w, h = fallback
        
        return (w, h)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # 1. 加载JSON并解析树结构
        with open(sample['json'], 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        # 🔥 关键：提取该JSON自己的分辨率（如1440x2560或1440x2392）
        json_w, json_h = self._get_json_resolution(json_data)

        # 🔥 防御：确保JSON分辨率有效
        if json_w <= 0 or json_h <= 0:
            json_w, json_h = 1440, 2560

        # 2. 加载图像
        img = Image.open(sample['image']).convert('RGB')
        actual_w, actual_h = img.size  # 可能是(1080,1920)

        # 🔥 关键：如果图片尺寸与JSON标注分辨率不一致，先对齐到JSON分辨率
        if (actual_w, actual_h) != (json_w, json_h):
            # 计算缩放因子，保持比例
            scale = min(json_w / actual_w, json_h / actual_h)
            new_w, new_h = int(actual_w * scale), int(actual_h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
            
            # 创建画布并居中粘贴（避免变形）
            new_img = Image.new('RGB', (json_w, json_h), (128, 128, 128))  # 灰色背景
            paste_x = (json_w - new_w) // 2
            paste_y = (json_h - new_h) // 2
            new_img.paste(img, (paste_x, paste_y))
            img = new_img

        img_tensor = self.transform(img)  # [3, H, W]
        img_tensor = img_tensor.permute(1, 2, 0)  # [H, W, 3]

        nodes, parent_map_str = parse_json_tree(json_data)
        
        # 3. 生成层级掩码（在latent空间64x64）
        depth_map, ancestry_map, parent_dict = create_maps_from_ui_tree(
            nodes, 
            img_wh=(json_w, json_h),
            latent_size=self.latent_size
        )
        
        # 4. 准备文本（caption）
        caption = sample['text']

        return {
            'image': img_tensor,           # [3, 256, 256]
            'caption': caption,            # str
            'depth_map': depth_map,        # [1, 1, 64, 64]
            'ancestry_map': ancestry_map,  # [1, 1, 64, 64]
            'parent_map': parent_dict,     # dict {child_id: parent_id}
            'filename': sample['id']       # 用于调试
        }
    # 在 dataset.py 的 UITreeDataset 类中添加
    @staticmethod
    def collate_fn(batch):
        """
        batch: List[Dict] 长度 = batch_size
        返回: Dict 所有字段都已批处理
        """
        # 1. 图像和张量字段直接 stack（这些都有相同的 tensor 结构）
        image        = torch.stack([item['image']        for item in batch])
        depth_map    = torch.stack([item['depth_map']    for item in batch])
        ancestry_map = torch.stack([item['ancestry_map'] for item in batch])
        
        # 2. 字符串字段保持为 List[str]（不能 stack）
        caption  = [item['caption']  for item in batch]
        filename = [item['filename'] for item in batch]
        parent_map = [item['parent_map'] for item in batch]
        
        return {
            'image':        image,        # [B, 3, 256, 256]
            'caption':      caption,      # List[str] 长度 B
            'depth_map':    depth_map,    # [B, 1, 64, 64]
            'ancestry_map': ancestry_map, # [B, 1, 64, 64]
            'parent_map':   parent_map,   # List[Dict] 长度 B
            'filename':     filename      # List[str] 长度 B
        }


# 兼容旧接口的包装类（如果你之前用了UIDataset和UITextDataset）
class UIDataset(UITreeDataset):
    """兼容旧接口，只返回image"""
    def __init__(self, root_dir, **kwargs):
        # 假设root_dir下同时有png和json
        super().__init__(
            image_dir=root_dir,
            json_dir=root_dir,
            text_dir=None,
            **kwargs
        )
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return {'image': data['image']}


class UITextDataset(UITreeDataset):
    """兼容旧接口，主要返回text（用于验证JSON解析）"""
    def __init__(self, json_dir: str, text_dir: str, **kwargs):
        super().__init__(
            image_dir="",  # 空，不加载图像
            json_dir=json_dir,
            text_dir=text_dir,
            **kwargs
        )
    
    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return {'text': data['caption']}