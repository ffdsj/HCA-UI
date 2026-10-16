import os
import json
import csv
import math
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from tqdm import tqdm


# ==================== JSON 解析 ====================

def parse_json_tree(data: dict) -> Tuple[List[dict], Dict[str, str], Tuple[int, int]]:
    """
    返回 (nodes, parent_map, img_wh)
    nodes: [{'id': str, 'bounds': [x1,y1,x2,y2], 'depth': int, 'class': str}, ...]
    parent_map: {child_id: parent_id}
    """
    if isinstance(data, dict):
        if 'activity' in data and isinstance(data['activity'], dict) and 'root' in data['activity']:
            root = data['activity']['root']
        elif 'root' in data:
            root = data['root']
        else:
            root = data
    else:
        raise ValueError("Invalid JSON")

    nodes: List[dict] = []
    parent_map: Dict[str, str] = {}
    counter = [0]

    # ========== 在 parse_json_tree 内，dfs 之前加 ==========
    SKIP_SYSTEM_CLASSES = {
        'com.android.internal.policy.PhoneWindow$DecorView',
        'android.view.ViewStub',
        'android.widget.Space',
    }

    def dfs(node, parent_id, depth, prefix):
        if node is None or not isinstance(node, dict):
            return ""
        bounds = node.get('bounds', [0, 0, 0, 0])
        if len(bounds) != 4 or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            return ""

        cls = node.get('class', '')
        visibility = node.get('visibility', 'visible')
        draw = node.get('draw', True)

        # ---------- 过滤：不可见 / 不绘制 / 系统容器 ----------
        should_skip = (
            visibility in ('gone', 'invisible') or
            draw is False or
            cls in SKIP_SYSTEM_CLASSES
        )

        if should_skip:
            # 跳过当前节点，children 顶替上来（parent_id、depth、prefix 都不变）
            for child in node.get('children', []):
                dfs(child, parent_id, depth, prefix)
            return ""
        # ------------------------------------------------------

        # 生成 seq（depth==0 也用 counter，防止根被跳过后子节点冲突）
        if depth == 0:
            if len(counter) == 0:
                counter.append(0)
            counter[0] += 1
            seq = str(counter[0])
        else:
            while len(counter) <= depth:
                counter.append(0)
            counter[depth] += 1
            seq = f"{prefix}-{counter[depth]}"

        nodes.append({
            'id': seq,
            'depth': depth,
            'parent_id': parent_id,
            'bounds': [int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3])],
            'class': cls or 'View',
        })

        for child in node.get('children', []):
            child_id = dfs(child, seq, depth + 1, seq)
            if child_id:
                parent_map[child_id] = seq
        return seq

    dfs(root, None, 0, "")
    root_bounds = root.get('bounds', [0,0,0,0])
    img_wh = (root_bounds[2], root_bounds[3])
    return nodes, parent_map, img_wh


# ==================== 几何工具 ====================

def bbox_area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

def bbox_iou(b1: List[float], b2: List[float]) -> float:
    x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = bbox_area(b1) + bbox_area(b2) - inter
    return inter / union if union > 0 else 0.0

def bbox_contains(outer: List[float], inner: List[float], margin: float = 0.0) -> bool:
    """Containment on normalized [0, 1] coordinates.

    margin uses the same normalized coordinate system. Default 0.0 avoids the
    previous unit mismatch where a pixel-scale margin was applied after normalization.
    """
    return (outer[0] - margin <= inner[0] and outer[1] - margin <= inner[1] and
            outer[2] + margin >= inner[2] and outer[3] + margin >= inner[3])

def hungarian_match(cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (row_idx, col_idx)"""
    try:
        from scipy.optimize import linear_sum_assignment
        return linear_sum_assignment(cost_matrix)
    except ImportError:
        # 贪心回退
        rows, cols = [], []
        cost = cost_matrix.copy()
        for _ in range(min(cost_matrix.shape)):
            r, c = np.unravel_index(np.argmin(cost), cost.shape)
            rows.append(r); cols.append(c)
            cost[r, :] = 1e9; cost[:, c] = 1e9
        return np.array(rows), np.array(cols)


# ==================== 新增：层级辅助函数 ====================

def is_ancestor(node_id: str, ancestor_id: str, parent_map: Dict[str, str]) -> bool:
    """判断 node_id 是否是 ancestor_id 的后代（用于 Overlap/Alignment 排除父子对）"""
    current = parent_map.get(node_id)
    while current is not None:
        if current == ancestor_id:
            return True
        current = parent_map.get(current)
    return False


# ==================== 核心评估类 ====================

class StructuralEvaluator:
    def __init__(self, iou_match_thresh: float = 0.3, containment_eps: float = 0.0):
        self.iou_match_thresh = iou_match_thresh
        self.containment_eps = containment_eps

    def evaluate_pair(self, gt_json_path: str, pred_json_path: str) -> Dict:
        """
        评估单对 GT/Pred JSON，返回所有结构指标。
        """
        with open(gt_json_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(pred_json_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)

        gt_nodes, gt_parent_map, gt_wh = parse_json_tree(gt_data)
        pred_nodes, pred_parent_map, pred_wh = parse_json_tree(pred_data)

        if gt_wh[0] <= 0 or gt_wh[1] <= 0:
            raise ValueError(f"Invalid GT image size: {gt_wh}")
        if pred_wh[0] <= 0 or pred_wh[1] <= 0:
            raise ValueError(f"Invalid Pred image size: {pred_wh}")

        # 归一化到 [0, 1]
        for n in gt_nodes:
            b = n['bounds']
            n['bounds'] = [b[0]/gt_wh[0], b[1]/gt_wh[1], b[2]/gt_wh[0], b[3]/gt_wh[1]]
        for n in pred_nodes:
            b = n['bounds']
            n['bounds'] = [b[0]/pred_wh[0], b[1]/pred_wh[1], b[2]/pred_wh[0], b[3]/pred_wh[1]]

        # 建立 ID -> node 映射
        gt_id2node = {n['id']: n for n in gt_nodes}
        pred_id2node = {n['id']: n for n in pred_nodes}

        # ========== 分层匹配：先父后子 ==========
        gt_id2node = {n['id']: n for n in gt_nodes}
        pred_id2node = {n['id']: n for n in pred_nodes}
        n_gt, n_pred = len(gt_nodes), len(pred_nodes)   # <-- 加这行
        if n_gt == 0 or n_pred == 0:
            return self._empty_result(n_gt, n_pred)

        # 父 -> 子列表
        gt_children = defaultdict(list)
        for cid, pid in gt_parent_map.items():
            gt_children[pid].append(cid)
        pred_children = defaultdict(list)
        for cid, pid in pred_parent_map.items():
            pred_children[pid].append(cid)

        gt2pred: Dict[str, str] = {}
        pred2gt: Dict[str, str] = {}
        matched_ious: List[float] = []

        def match_level(gt_ids: List[str], pred_ids: List[str]):
            """在同一父节点下做匈牙利匹配，匹配上的递归进子树"""
            if not gt_ids or not pred_ids:
                return
            n_g, n_p = len(gt_ids), len(pred_ids)
            cost = np.zeros((n_g, n_p))
            for i, gid in enumerate(gt_ids):
                for j, pid in enumerate(pred_ids):
                    cost[i, j] = 1.0 - bbox_iou(gt_id2node[gid]['bounds'], pred_id2node[pid]['bounds'])
            rows, cols = hungarian_match(cost)
            for r, c in zip(rows, cols):
                iou = 1.0 - cost[r, c]
                if iou >= self.iou_match_thresh:
                    gid, pid = gt_ids[r], pred_ids[c]
                    gt2pred[gid] = pid
                    pred2gt[pid] = gid
                    matched_ious.append(iou)
                    # 递归子树
                    match_level(gt_children.get(gid, []), pred_children.get(pid, []))

        # 从根开始
        gt_roots = [n['id'] for n in gt_nodes if n['parent_id'] is None]
        pred_roots = [n['id'] for n in pred_nodes if n['parent_id'] is None]
        match_level(gt_roots, pred_roots)

        # 兜底：未匹配的节点全局再配一次（防止父节点没对上导致子树全丢）
        unmatched_gt = [n['id'] for n in gt_nodes if n['id'] not in gt2pred]
        unmatched_pred = [n['id'] for n in pred_nodes if n['id'] not in pred2gt]
        if unmatched_gt and unmatched_pred:
            n_ug, n_up = len(unmatched_gt), len(unmatched_pred)
            cost = np.zeros((n_ug, n_up))
            for i, gid in enumerate(unmatched_gt):
                for j, pid in enumerate(unmatched_pred):
                    cost[i, j] = 1.0 - bbox_iou(gt_id2node[gid]['bounds'], pred_id2node[pid]['bounds'])
            rows, cols = hungarian_match(cost)
            for r, c in zip(rows, cols):
                iou = 1.0 - cost[r, c]
                if iou >= self.iou_match_thresh:
                    gt2pred[unmatched_gt[r]] = unmatched_pred[c]
                    pred2gt[unmatched_pred[c]] = unmatched_gt[r]
                    matched_ious.append(iou)
        bbi = (
            float(np.sum(matched_ious)) / max(n_gt, n_pred)
            if max(n_gt, n_pred) > 0
            else 0.0
        )
        # =========================================

        # ---------- 标准 LayoutGAN++ Alignment（越低越好） ----------
        alignment = self._alignment(pred_nodes)

        # ---------- RQ1: 整体框架 ----------
        # 保留 layout_alignment 键仅为兼容旧的 CSV/绘图脚本；
        # 其数值现在等同于标准 LayoutGAN++ Alignment，不再使用自定义 IoU*coverage。
        layout_alignment = alignment
        struct_pass = self._structural_constraint_pass_rate(pred_nodes, pred_parent_map)
        repair = self._repair_metrics(gt_nodes, pred_nodes, gt2pred, pred2gt)

        # ---------- RQ2: 层级约束 ----------
        pcp = self._parent_child_preservation(gt_nodes, gt_parent_map, pred_nodes, pred_parent_map, gt2pred, pred2gt)
        npl = self._non_parent_leakage(gt_nodes, gt_parent_map, pred_nodes, pred_parent_map, gt2pred, pred2gt)
        cleanliness = self._hierarchical_cleanliness(pcp, struct_pass['sibling_conflict_rate'], gt_nodes, pred_nodes, gt2pred)

        # ---------- RQ3: 自适应 Gate / 深度 ----------
        depth_acc = self._depth_accuracy(gt_nodes, pred_nodes, gt2pred)
        small_recall = self._small_component_recall(gt_nodes, pred_nodes, gt2pred)
        num_small_gt = self._num_small_components(gt_nodes)

        # ---------- RQ4 ----------
        overlap = self._overlap(pred_nodes, pred_parent_map)
        cvr = self._constraint_violation_rate(pred_nodes, pred_parent_map)
        svr = self._sibling_conflict_rate(pred_nodes, pred_parent_map)
        bvr = self._boundary_violation_rate(pred_nodes, pred_parent_map)
        mcs = self._multi_child_stability(gt_nodes, gt_parent_map, pred_nodes, pred_parent_map, gt2pred, pred2gt)

        # ---------- 通用 ----------
        node_f1 = self._node_count_f1(n_gt, n_pred, len(gt2pred))
        area_ratio_preservation = self._area_ratio_preservation(gt_nodes, gt_parent_map, pred_nodes, pred_parent_map, gt2pred)

        return {
            'RQ1': {
                'layout_alignment': layout_alignment,
                'structural_constraint_pass_rate': struct_pass['pass_rate'],
                'repair_rate': repair['repair_rate'],
                'minimal_edit_distance': repair['edit_distance'],
                'patch_size_ratio': repair['patch_size_ratio'],
            },
            'RQ2': {
                'parent_child_preservation_rate': pcp,
                'non_parent_leakage_rate': npl,
                'hierarchical_cleanliness_score': cleanliness,
            },
            'RQ3': {
                'depth_accuracy': depth_acc,
                'small_component_recall': small_recall,
            },
            'RQ4': {
                'overlap': overlap,
                'alignment': alignment,
                'constraint_violation_rate': cvr,
                'sibling_conflict_rate': svr,
                'boundary_violation_rate': bvr,
                'multi_child_stability_score': mcs,
            },
            'common': {
                'node_count_precision': node_f1['precision'],
                'node_count_recall': node_f1['recall'],
                'node_count_f1': node_f1['f1'],
                'num_gt_nodes': n_gt,
                'num_pred_nodes': n_pred,
                'num_matched_nodes': len(gt2pred),
                'bounding_box_iou': bbi,
                'area_ratio_preservation': area_ratio_preservation,
                'num_small_gt': num_small_gt,
            }
        }

    # ------------------------------------------------------------------
    # 子指标实现
    # ------------------------------------------------------------------

    def _structural_constraint_pass_rate(self, pred_nodes: List[dict], pred_parent_map: Dict[str, str]) -> Dict:
        id2node = {n['id']: n for n in pred_nodes}
        children_of = defaultdict(list)
        for cid, pid in pred_parent_map.items():
            children_of[pid].append(cid)

        total = 0
        passed = 0

        # a) 父子包含
        for cid, pid in pred_parent_map.items():
            if pid not in id2node or cid not in id2node:
                continue
            total += 1
            if bbox_contains(id2node[pid]['bounds'], id2node[cid]['bounds'], self.containment_eps):
                passed += 1

        # b) 兄弟不重叠
        sibling_conflicts = 0
        sibling_pairs = 0
        for pid, cids in children_of.items():
            cids = [c for c in cids if c in id2node]
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    sibling_pairs += 1
                    if bbox_iou(id2node[cids[i]]['bounds'], id2node[cids[j]]['bounds']) > 0.0:
                        sibling_conflicts += 1
                    else:
                        passed += 1
                    total += 1

        # c) 有效面积
        for n in pred_nodes:
            total += 1
            if bbox_area(n['bounds']) > 0:
                passed += 1

        return {
            'pass_rate': passed / total if total else 0.0,
            'sibling_conflict_rate': sibling_conflicts / sibling_pairs if sibling_pairs else 0.0,
        }

    def _repair_metrics(self, gt_nodes, pred_nodes, gt2pred, pred2gt):
        n_gt, n_pred = len(gt_nodes), len(pred_nodes)
        n_matched = len(gt2pred)

        to_add = n_gt - n_matched
        to_del = n_pred - n_matched

        # 需要修改 bbox 的匹配节点（IoU < 0.9）
        to_edit = 0
        patch_area = 0
        gt_id2node = {n['id']: n for n in gt_nodes}
        pred_id2node = {n['id']: n for n in pred_nodes}

        for gid, pid in gt2pred.items():
            g, p = gt_id2node[gid], pred_id2node[pid]
            iou = bbox_iou(g['bounds'], p['bounds'])
            if iou < 0.9:
                to_edit += 1
                # 对称差面积近似
                ga, pa = bbox_area(g['bounds']), bbox_area(p['bounds'])
                patch_area += ga + pa - 2 * iou * (ga + pa) / 2

        edit_distance = to_add + to_del + to_edit
        repair_rate = edit_distance / n_gt if n_gt else 0.0
        # 归一化 patch 面积到全图（假设 1440x2560）
        patch_size_ratio = patch_area / 1.0

        return {
            'repair_rate': repair_rate,
            'edit_distance': edit_distance,
            'patch_size_ratio': patch_size_ratio,
            'to_add': to_add,
            'to_del': to_del,
            'to_edit': to_edit,
        }

    def _parent_child_preservation(self, gt_nodes, gt_parent_map, pred_nodes, pred_parent_map, gt2pred, pred2gt):
        """Parent-Child Preservation (PCP).

        Denominator = all valid GT parent-child edges. If either endpoint is not
        matched, that GT edge is counted as not preserved rather than removed from
        the denominator. A matched edge is preserved when either:
          1) the predicted child has the mapped predicted parent explicitly; or
          2) the mapped child is spatially contained by the mapped parent.
        """
        gt_id2node = {n['id']: n for n in gt_nodes}
        pred_id2node = {n['id']: n for n in pred_nodes}

        preserved = 0
        total = 0
        for cid, pid in gt_parent_map.items():
            if cid not in gt_id2node or pid not in gt_id2node:
                continue
            total += 1

            # 漏检 parent 或 child：该 GT edge 直接计为未保持
            if cid not in gt2pred or pid not in gt2pred:
                continue

            pred_cid = gt2pred[cid]
            pred_pid = gt2pred[pid]
            if pred_cid not in pred_id2node or pred_pid not in pred_id2node:
                continue

            actual_pred_pid = pred_parent_map.get(pred_cid)
            if actual_pred_pid == pred_pid:
                preserved += 1
            elif bbox_contains(
                pred_id2node[pred_pid]['bounds'],
                pred_id2node[pred_cid]['bounds'],
                self.containment_eps
            ):
                preserved += 1

        return preserved / total if total else 0.0

    def _non_parent_leakage(self, gt_nodes, gt_parent_map, pred_nodes, pred_parent_map, gt2pred, pred2gt):
        gt_id2node = {n['id']: n for n in gt_nodes}
        pred_id2node = {n['id']: n for n in pred_nodes}

        leakage = 0
        total = 0
        for cid, pid in gt_parent_map.items():
            if cid not in gt2pred or pid not in gt2pred:
                continue
            total += 1
            pred_cid = gt2pred[cid]
            pred_pid = gt2pred[pid]
            actual_pred_pid = pred_parent_map.get(pred_cid)
            if actual_pred_pid != pred_pid:
                # 进一步检查：actual_pred_pid 对应的 GT 节点是不是 cid 的祖先
                actual_gt_pid = pred2gt.get(actual_pred_pid)
                if actual_gt_pid != pid:
                    leakage += 1

        return leakage / total if total else 0.0

    def _hierarchical_cleanliness(self, pcp, sibling_conflict_rate, gt_nodes, pred_nodes, gt2pred):
        # 深度准确率
        gt_id2node = {n['id']: n for n in gt_nodes}
        pred_id2node = {n['id']: n for n in pred_nodes}
        depth_accs = []
        for gid, pid in gt2pred.items():
            d_gt = gt_id2node[gid]['depth']
            d_pred = pred_id2node[pid]['depth']
            depth_accs.append(1.0 - abs(d_gt - d_pred) / max(d_gt, 1))
        da = float(np.mean(depth_accs)) if depth_accs else 0.0
        return 0.5 * pcp + 0.3 * (1.0 - sibling_conflict_rate) + 0.2 * da

    def _depth_accuracy(self, gt_nodes, pred_nodes, gt2pred):
        gt_id2node = {n['id']: n for n in gt_nodes}
        pred_id2node = {n['id']: n for n in pred_nodes}
        accs = []
        for gid, pid in gt2pred.items():
            d_gt = gt_id2node[gid]['depth']
            d_pred = pred_id2node[pid]['depth']
            accs.append(1.0 - abs(d_gt - d_pred) / max(d_gt, 1))
        return float(np.mean(accs)) if accs else 0.0

    def _num_small_components(self, gt_nodes) -> int:
        """GT small components: non-root nodes with normalized area < 1%."""
        return sum(
            1 for n in gt_nodes
            if n['depth'] > 0 and bbox_area(n['bounds']) < 0.01
        )

    def _small_component_recall(self, gt_nodes, pred_nodes, gt2pred):
        """Small Component Recall.

        GT components with normalized area < 0.01 and depth > 0 are small.
        Pages without any GT small component return NaN and are excluded from the
        macro mean; batch summary reports valid_n for this metric.
        """
        small_gt = [
            n for n in gt_nodes
            if n['depth'] > 0 and bbox_area(n['bounds']) < 0.01
        ]
        if not small_gt:
            return np.nan
        matched = sum(1 for n in small_gt if n['id'] in gt2pred)
        return matched / len(small_gt)

    def _sibling_conflict_rate(self, pred_nodes, pred_parent_map):
        id2node = {n['id']: n for n in pred_nodes}
        children_of = defaultdict(list)
        for cid, pid in pred_parent_map.items():
            children_of[pid].append(cid)

        conflicts = 0
        pairs = 0
        for pid, cids in children_of.items():
            cids = [c for c in cids if c in id2node]
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    pairs += 1
                    if bbox_iou(id2node[cids[i]]['bounds'], id2node[cids[j]]['bounds']) > 0.0:
                        conflicts += 1
        return conflicts / pairs if pairs else 0.0

    def _boundary_violation_rate(self, pred_nodes, pred_parent_map):
        id2node = {n['id']: n for n in pred_nodes}
        violations = 0
        total = 0
        for cid, pid in pred_parent_map.items():
            if cid not in id2node or pid not in id2node:
                continue
            total += 1
            if not bbox_contains(id2node[pid]['bounds'], id2node[cid]['bounds'], self.containment_eps):
                violations += 1
        return violations / total if total else 0.0

    def _overlap(self, pred_nodes: List[dict], pred_parent_map: Dict[str, str]) -> float:
        """
        LayoutGAN++ 标准 Overlap 指标（适配层级结构）。
        排除祖先-后代关系对（父子包含是正常层级行为），
        计算其余节点对的平均重叠比例。
        定义参考: Kikuchi et al., MM 2021
        """
        n = len(pred_nodes)
        if n < 2:
            return 0.0

        id2node = {n['id']: n for n in pred_nodes}
        ids = [n['id'] for n in pred_nodes]

        total = 0.0
        for i in range(n):
            b1 = id2node[ids[i]]['bounds']
            area1 = bbox_area(b1)
            if area1 == 0:
                continue

            for j in range(n):
                if i == j:
                    continue
                # 排除祖先-后代关系（层级包含不是重叠违规）
                if is_ancestor(ids[i], ids[j], pred_parent_map) or is_ancestor(ids[j], ids[i], pred_parent_map):
                    continue

                b2 = id2node[ids[j]]['bounds']
                inter = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0])) * \
                        max(0, min(b1[3], b2[3]) - max(b1[1], b2[1]))
                total += inter / area1

        return total / n if n > 0 else 0.0

    def _alignment(self, pred_nodes: List[dict]) -> float:
        """Standard LayoutGAN++ Alignment loss (lower is better).

        Equivalent to the official compute_alignment implementation. For each
        element, find the minimum normalized distance to another element among six
        same-type alignment lines: left, x-center, right, top, y-center, bottom.
        Convert that distance with -log(1-d), then average over elements.

        IMPORTANT: this is the standard flat-layout metric, so ancestor/descendant
        pairs are NOT removed here.
        """
        n = len(pred_nodes)
        if n < 2:
            return 0.0

        boxes = [n_['bounds'] for n_ in pred_nodes]
        total = 0.0

        for i, b1 in enumerate(boxes):
            l1, t1, r1, btm1 = b1
            xc1 = (l1 + r1) / 2.0
            yc1 = (t1 + btm1) / 2.0

            best = 1.0
            for j, b2 in enumerate(boxes):
                if i == j:
                    continue
                l2, t2, r2, btm2 = b2
                xc2 = (l2 + r2) / 2.0
                yc2 = (t2 + btm2) / 2.0

                d = min(
                    abs(l1 - l2),
                    abs(xc1 - xc2),
                    abs(r1 - r2),
                    abs(t1 - t2),
                    abs(yc1 - yc2),
                    abs(btm1 - btm2),
                )
                best = min(best, d)

            # Normalized coordinates should keep best in [0,1]; clip only for safety.
            best = min(max(float(best), 0.0), 1.0 - 1e-12)
            total += -math.log(1.0 - best)

        return total / n

    def _constraint_violation_rate(self, pred_nodes: List[dict], pred_parent_map: Dict[str, str]) -> float:
        """
        综合约束违反率（Unified Constraint Violation Rate）。
        整合：(1) 父子包含违反 (2) 兄弟重叠违反 (3) 无效面积。
        与 LayoutGAN++ / LayoutDM 的 Constraint Violation 框架对齐。
        """
        id2node = {n['id']: n for n in pred_nodes}
        children_of = defaultdict(list)
        for cid, pid in pred_parent_map.items():
            children_of[pid].append(cid)

        total_checks = 0
        violations = 0

        # (1) 父子包含约束：子必须严格在父内部
        for cid, pid in pred_parent_map.items():
            if pid not in id2node or cid not in id2node:
                continue
            total_checks += 1
            if not bbox_contains(id2node[pid]['bounds'], id2node[cid]['bounds'], self.containment_eps):
                violations += 1

        # (2) 兄弟不重叠约束
        for pid, cids in children_of.items():
            cids = [c for c in cids if c in id2node]
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    total_checks += 1
                    if bbox_iou(id2node[cids[i]]['bounds'], id2node[cids[j]]['bounds']) > 0.0:
                        violations += 1

        # (3) 有效面积约束
        for n in pred_nodes:
            total_checks += 1
            if bbox_area(n['bounds']) <= 0:
                violations += 1

        return violations / total_checks if total_checks > 0 else 0.0

    def _multi_child_stability(self, gt_nodes, gt_parent_map, pred_nodes, pred_parent_map, gt2pred, pred2gt):
        # 构建父->子
        gt_children_of = defaultdict(list)
        for cid, pid in gt_parent_map.items():
            gt_children_of[pid].append(cid)
        pred_children_of = defaultdict(list)
        for cid, pid in pred_parent_map.items():
            pred_children_of[pid].append(cid)

        gt_id2node = {n['id']: n for n in gt_nodes}
        pred_id2node = {n['id']: n for n in pred_nodes}

        # 匹配父节点
        gt_parents = list(gt_children_of.keys())
        pred_parents = list(pred_children_of.keys())
        n_gp, n_pp = len(gt_parents), len(pred_parents)
        if n_gp == 0 or n_pp == 0:
            return 0.0

        cost = np.zeros((n_gp, n_pp))
        for i, gpid in enumerate(gt_parents):
            for j, ppid in enumerate(pred_parents):
                g = gt_id2node.get(gpid)
                p = pred_id2node.get(ppid)
                if g is None or p is None:
                    cost[i, j] = 1.0
                else:
                    cost[i, j] = 1.0 - bbox_iou(g['bounds'], p['bounds'])
        rows, cols = hungarian_match(cost)
        parent_match = {gt_parents[r]: pred_parents[c] for r, c in zip(rows, cols)
                        if 1.0 - cost[r, c] > 0.1}

        scores = []
        for gpid, ppid in parent_match.items():
            gt_cids = [c for c in gt_children_of[gpid] if c in gt_id2node]
            pred_cids = [c for c in pred_children_of[ppid] if c in pred_id2node]
            if len(gt_cids) < 2 or len(pred_cids) < 2:
                continue

            # 子节点匹配
            n1, n2 = len(gt_cids), len(pred_cids)
            c_cost = np.zeros((n1, n2))
            for i, gc in enumerate(gt_cids):
                for j, pc in enumerate(pred_cids):
                    c_cost[i, j] = 1.0 - bbox_iou(gt_id2node[gc]['bounds'], pred_id2node[pc]['bounds'])
            r2, c2 = hungarian_match(c_cost)
            matched = [(gt_cids[i], pred_cids[j]) for i, j in zip(r2, c2) if 1.0 - c_cost[i, j] > 0.1]
            if len(matched) < 2:
                continue

            # 相对位置矩阵的相关系数
            def centers(nodes, ids):
                return np.array([
                    [(nodes[i]['bounds'][0] + nodes[i]['bounds'][2]) / 2,
                     (nodes[i]['bounds'][1] + nodes[i]['bounds'][3]) / 2]
                    for i in ids
                ])

            gt_centers = centers(gt_id2node, [m[0] for m in matched])
            pred_centers = centers(pred_id2node, [m[1] for m in matched])

            # 归一化到父节点尺寸
            gb = gt_id2node[gpid]['bounds']
            pb = pred_id2node[ppid]['bounds']
            gt_wh = np.array([gb[2] - gb[0], gb[3] - gb[1]])
            pred_wh = np.array([pb[2] - pb[0], pb[3] - pb[1]])
            if np.any(gt_wh == 0) or np.any(pred_wh == 0):
                continue

            gt_norm = (gt_centers - np.array([gb[0], gb[1]])) / gt_wh
            pred_norm = (pred_centers - np.array([pb[0], pb[1]])) / pred_wh

            # 距离矩阵
            gt_dist = np.linalg.norm(gt_norm[:, None] - gt_norm[None, :], axis=2)
            pred_dist = np.linalg.norm(pred_norm[:, None] - pred_norm[None, :], axis=2)

            if gt_dist.size > 1:
                # 避免 std=0 导致 divide by zero 警告
                if np.std(gt_dist) > 1e-9 and np.std(pred_dist) > 1e-9:
                    corr = np.corrcoef(gt_dist.flatten(), pred_dist.flatten())[0, 1]
                    if not np.isnan(corr) and not np.isinf(corr):
                        scores.append(max(0.0, corr))

        return float(np.mean(scores)) if scores else 0.0

    def _node_count_f1(self, n_gt, n_pred, n_matched):
        precision = n_matched / n_pred if n_pred else 0.0
        recall = n_matched / n_gt if n_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def _area_ratio_preservation(self, gt_nodes, gt_parent_map, pred_nodes, pred_parent_map, gt2pred):
        """父子面积比的保持程度"""
        gt_id2node = {n['id']: n for n in gt_nodes}
        pred_id2node = {n['id']: n for n in pred_nodes}

        ratios = []
        for cid, pid in gt_parent_map.items():
            if cid not in gt2pred or pid not in gt2pred:
                continue
            pred_cid = gt2pred[cid]
            pred_pid = gt2pred[pid]

            gt_ratio = bbox_area(gt_id2node[cid]['bounds']) / max(bbox_area(gt_id2node[pid]['bounds']), 1)
            pred_ratio = bbox_area(pred_id2node[pred_cid]['bounds']) / max(bbox_area(pred_id2node[pred_pid]['bounds']), 1)

            # 相对误差
            if max(gt_ratio, pred_ratio) > 0:
                err = abs(gt_ratio - pred_ratio) / max(gt_ratio, pred_ratio)
                ratios.append(1.0 - err)

        return float(np.mean(ratios)) if ratios else 0.0


    def _empty_result(self, n_gt, n_pred):
        return {
            'RQ1': {
                'layout_alignment': np.nan,
                'structural_constraint_pass_rate': 0.0,
                'repair_rate': 1.0,
                'minimal_edit_distance': max(n_gt, n_pred),
                'patch_size_ratio': 0.0,
            },
            'RQ2': {
                'parent_child_preservation_rate': 0.0,
                'non_parent_leakage_rate': 0.0,
                'hierarchical_cleanliness_score': 0.0,
            },
            'RQ3': {
                'depth_accuracy': 0.0,
                'small_component_recall': 0.0,
            },
            'RQ4': {
                'overlap': 0.0,
                'alignment': np.nan,
                'constraint_violation_rate': 0.0,
                'sibling_conflict_rate': 0.0,
                'boundary_violation_rate': 0.0,
                'multi_child_stability_score': 0.0,
            },
            'common': {
                'node_count_precision': 0.0,
                'node_count_recall': 0.0,
                'node_count_f1': 0.0,
                'num_gt_nodes': n_gt,
                'num_pred_nodes': n_pred,
                'num_matched_nodes': 0,
                'mean_matched_iou': 0.0,
                'area_ratio_preservation': 0.0,
                'global_overlap_rate': 0.0,
                'num_small_gt': 0,
            }
        }


# ==================== 批量评估 CLI ====================

def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _sort_id(x: str):
    return (0, int(x)) if x.isdigit() else (1, x)


def _load_sample_ids(sample_ids_file: Optional[str]) -> Optional[List[str]]:
    if not sample_ids_file:
        return None
    p = Path(sample_ids_file)
    if not p.exists():
        raise FileNotFoundError(sample_ids_file)
    if p.suffix.lower() == '.json':
        data = json.loads(p.read_text(encoding='utf-8'))
        if isinstance(data, dict):
            for key in ('ids', 'sample_ids', 'pages'):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        if not isinstance(data, list):
            raise ValueError('sample_ids JSON must be a list or contain ids/sample_ids/pages list')
        return [str(x) for x in data]
    return [line.strip() for line in p.read_text(encoding='utf-8').splitlines() if line.strip()]


def batch_evaluate(
    gt_json_dir: str,
    pred_json_dir: str,
    output_csv: str,
    iou_thresh: float = 0.3,
    containment_eps: float = 0.0,
    max_samples: Optional[int] = None,
    sample_ids_file: Optional[str] = None,
):
    """Batch evaluation without silently dropping failures.

    The GT set defines the denominator. Missing/failed predictions remain in the CSV.
    For a failed sample, PCP and SCR are set to 0 as recall/preservation failures;
    metrics that cannot be defined are NaN.
    """
    evaluator = StructuralEvaluator(
        iou_match_thresh=iou_thresh,
        containment_eps=containment_eps,
    )

    gt_files = {Path(f).stem: os.path.join(gt_json_dir, f)
                for f in os.listdir(gt_json_dir) if f.endswith('.json')}
    pred_files = {Path(f).stem: os.path.join(pred_json_dir, f)
                  for f in os.listdir(pred_json_dir) if f.endswith('.json')}

    requested_ids = _load_sample_ids(sample_ids_file)
    if requested_ids is None:
        eval_ids = sorted(gt_files.keys(), key=_sort_id)
    else:
        eval_ids = [sid for sid in requested_ids if sid in gt_files]

    if max_samples is not None and max_samples > 0:
        eval_ids = eval_ids[:max_samples]

    print(f"GT: {len(gt_files)}, Pred: {len(pred_files)}, Evaluating: {len(eval_ids)}")

    rows = []
    failures = 0
    for stem in tqdm(eval_ids, desc="Evaluating"):
        if stem not in pred_files:
            failures += 1
            rows.append({
                'id': stem,
                'parse_success': 0,
                'failure_reason': 'missing_prediction_file',
                'RQ1.layout_alignment': np.nan,
                'RQ2.parent_child_preservation_rate': 0.0,
                'RQ3.small_component_recall': 0.0,
                'RQ4.alignment': np.nan,
                'RQ4.constraint_violation_rate': np.nan,
                'common.mean_matched_iou': 0.0,
            })
            continue

        try:
            result = evaluator.evaluate_pair(gt_files[stem], pred_files[stem])
            flat = flatten_dict(result)
            flat['id'] = stem
            flat['parse_success'] = 1
            flat['failure_reason'] = ''
            rows.append(flat)
        except Exception as e:
            failures += 1
            rows.append({
                'id': stem,
                'parse_success': 0,
                'failure_reason': f'{type(e).__name__}: {e}',
                'RQ1.layout_alignment': np.nan,
                'RQ2.parent_child_preservation_rate': 0.0,
                'RQ3.small_component_recall': 0.0,
                'RQ4.alignment': np.nan,
                'RQ4.constraint_violation_rate': np.nan,
                'common.mean_matched_iou': 0.0,
            })

    if not rows:
        print("❌ 没有可评估样本")
        return

    # Union of keys, because failed rows may not contain all metrics.
    keys = ['id', 'parse_success', 'failure_reason']
    other_keys = sorted({k for row in rows for k in row.keys()} - set(keys))
    keys.extend(other_keys)

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅ 逐样本结果已保存: {output_csv}")

    import pandas as pd
    df = pd.read_csv(output_csv)
    metric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns
        if c not in ('parse_success',)
    ]
    summary_rows = []
    for c in metric_cols:
        summary_rows.append({
            'metric': c,
            'mean': df[c].mean(skipna=True),
            'valid_n': int(df[c].count()),
            'total_n': int(len(df)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_csv.replace('.csv', '_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"📊 汇总已保存: {summary_path}")
    print(f"Parser/missing failures: {failures}/{len(eval_ids)}")


def main():
    gt_dir = ""              
    pred_dir = ""          
    out_csv = ""      
    iou_thresh = 0.15                        
    containment_eps = 0.0                   
    max_samples = 1000
    sample_ids_file = None                  

    batch_evaluate(
        gt_dir, pred_dir, out_csv,
        iou_thresh=iou_thresh,
        containment_eps=containment_eps,
        max_samples=max_samples,
        sample_ids_file=sample_ids_file,
    )


if __name__ == "__main__":
    main()



import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from jsonschema import Draft202012Validator
from zss import Node, simple_distance


# ============================================================
# 1. JSON Schema
# ============================================================

UI_NODE_SCHEMA = {
    "type": "object",

    "properties": {
        "class": {
            "type": "string"
        },

        "bounds": {
            "type": "array",
            "minItems": 4,
            "maxItems": 4,
            "items": {
                "type": "number"
            }
        },

        "children": {
            "type": "array",
            "items": {
                "$ref": "#/$defs/node"
            }
        },

        "visibility": {
            "type": "string"
        },

        "draw": {
            "type": "boolean"
        },

        "resource-id": {
            "type": "string"
        },

        "content-desc": {},

        "clickable": {
            "type": "boolean"
        },

        "focusable": {
            "type": "boolean"
        },

        "enabled": {
            "type": "boolean"
        },

        "selected": {
            "type": "boolean"
        },

        "focused": {
            "type": "boolean"
        }
    },

    # 原版 JSON 有大量 Android 特有字段，
    # 所以这里不能 additionalProperties=False
    "additionalProperties": True,

    "$defs": {
        "node": {
            "type": "object",

            "properties": {
                "class": {
                    "type": "string"
                },

                "bounds": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {
                        "type": "number"
                    }
                },

                "children": {
                    "type": "array",
                    "items": {
                        "$ref": "#/$defs/node"
                    }
                }
            },

            "additionalProperties": True
        }
    }
}


ROOT_SCHEMA = {
    "type": "object",

    "properties": {
        "activity_name": {
            "type": "string"
        },

        "activity": {
            "type": "object",
            "required": ["root"],

            "properties": {
                "root": {
                    "$ref": "#/$defs/node"
                }
            },

            "required": ["root"],
            "additionalProperties": True
        },

        "root": {
            "$ref": "#/$defs/node"
        }
    },

    "$defs": {
        "node": UI_NODE_SCHEMA["$defs"]["node"]
    },

    # 允许 activity_name + activity 等其他字段
    "additionalProperties": True
}


# ============================================================
# 2. 找到 root
# ============================================================

def get_root(data: Dict[str, Any]) -> Dict[str, Any]:

    if not isinstance(data, dict):
        raise ValueError("JSON 顶层必须是 object")

    # 原版 Rico
    activity = data.get("activity")

    if isinstance(activity, dict):
        root = activity.get("root")

        if isinstance(root, dict):
            return root

    # 部分简化格式
    root = data.get("root")

    if isinstance(root, dict):
        return root

    # 最后认为整个 JSON 就是 root
    return data


# ============================================================
# 3. Schema Validation
# ============================================================

def validate_schema(data: Dict[str, Any]) -> Dict[str, Any]:

    validator = Draft202012Validator(ROOT_SCHEMA)

    errors = sorted(
        validator.iter_errors(data),
        key=lambda e: list(e.path)
    )

    return {
        "valid": len(errors) == 0,
        "num_errors": len(errors),
        "errors": [
            {
                "path": ".".join(str(x) for x in error.path),
                "message": error.message
            }
            for error in errors
        ]
    }


# ============================================================
# 4. 判断是否应该跳过节点
# ============================================================

SKIP_SYSTEM_CLASSES = {
    "com.android.internal.policy.PhoneWindow$DecorView",
    "android.view.ViewStub",
    "android.widget.Space",
}

SKIP_SYSTEM_RESOURCE_IDS = {
    "android:id/decor_content_parent",
    "android:id/content",
}


def should_skip(node: Dict[str, Any]) -> bool:

    # invisible / gone
    visibility = node.get("visibility", "visible")

    if visibility in ("gone", "invisible"):
        return True

    # draw=false
    if node.get("draw", True) is False:
        return True

    # 系统包装类
    cls = node.get("class", "")

    if cls in SKIP_SYSTEM_CLASSES:
        return True

    # 系统 resource-id
    resource_id = node.get("resource-id")

    if resource_id in SKIP_SYSTEM_RESOURCE_IDS:
        return True

    return False


# ============================================================
# 5. 节点标签
# ============================================================

def get_node_label(node: Dict[str, Any]) -> str:

    cls = node.get("class")

    if isinstance(cls, str) and cls.strip():
        return cls.strip()

    resource_id = node.get("resource-id")

    if isinstance(resource_id, str) and resource_id.strip():

        # Android 系统 ID 不作为语义节点
        if resource_id not in SKIP_SYSTEM_RESOURCE_IDS:
            return f"resource:{resource_id}"

    return "View"


# ============================================================
# 6. 构建统一 UI Tree
# ============================================================

def build_tree(node: Dict[str, Any]) -> Optional[Node]:

    if not isinstance(node, dict):
        return None

    # 当前节点需要跳过
    if should_skip(node):

        children = node.get("children", [])

        if not isinstance(children, list):
            return None

        valid_children = []

        for child in children:

            child_tree = build_tree(child)

            if child_tree is not None:
                valid_children.append(child_tree)

        # 一个被跳过的节点不能直接返回多个 Node
        # 因此这里需要特殊处理
        if len(valid_children) == 1:
            return valid_children[0]

        if len(valid_children) > 1:
            wrapper = Node("SKIPPED_WRAPPER")

            for child_tree in valid_children:
                wrapper.addkid(child_tree)

            return wrapper

        return None

    # 当前节点
    label = get_node_label(node)

    tree = Node(label)

    children = node.get("children", [])

    if not isinstance(children, list):
        children = []

    for child in children:

        child_tree = build_tree(child)

        if child_tree is not None:
            tree.addkid(child_tree)

    return tree


# ============================================================
# 7. 更严格的 Tree 构建版本
# ============================================================

def build_tree_flatten_skip(node: Dict[str, Any]) -> Optional[Node]:

    if not isinstance(node, dict):
        return None

    children = node.get("children", [])

    if not isinstance(children, list):
        children = []

    # 当前节点需要跳过
    if should_skip(node):

        # 如果只有一个有效 child
        valid_children = []

        for child in children:

            child_tree = build_tree_flatten_skip(child)

            if child_tree is not None:
                valid_children.append(child_tree)

        if len(valid_children) == 0:
            return None

        if len(valid_children) == 1:
            return valid_children[0]

        # 多个 child 时用一个透明 root，
        # 保持这些节点的顺序关系
        wrapper = Node("ROOT")

        for child_tree in valid_children:
            wrapper.addkid(child_tree)

        return wrapper

    # 正常节点
    tree = Node(get_node_label(node))

    for child in children:

        child_tree = build_tree_flatten_skip(child)

        if child_tree is not None:
            tree.addkid(child_tree)

    return tree


# ============================================================
# 8. Tree Size
# ============================================================

def tree_size(node: Optional[Node]) -> int:

    if node is None:
        return 0

    total = 1

    for child in node.children:
        total += tree_size(child)

    return total


# ============================================================
# 9. TED
# ============================================================

def calculate_ted(
    gt_data: Dict[str, Any],
    pred_data: Dict[str, Any]
) -> Dict[str, float]:

    gt_root = get_root(gt_data)
    pred_root = get_root(pred_data)

    gt_tree = build_tree_flatten_skip(gt_root)
    pred_tree = build_tree_flatten_skip(pred_root)

    # 空树
    if gt_tree is None and pred_tree is None:
        return {
            "ted": 0.0,
            "normalized_ted": 0.0,
            "gt_nodes": 0,
            "pred_nodes": 0
        }

    if gt_tree is None:
        pred_nodes = tree_size(pred_tree)

        return {
            "ted": float(pred_nodes),
            "normalized_ted": 1.0,
            "gt_nodes": 0,
            "pred_nodes": pred_nodes
        }

    if pred_tree is None:
        gt_nodes = tree_size(gt_tree)

        return {
            "ted": float(gt_nodes),
            "normalized_ted": 1.0,
            "gt_nodes": gt_nodes,
            "pred_nodes": 0
        }

    # Zhang-Shasha ordered tree edit distance
    ted = simple_distance(
        gt_tree,
        pred_tree
    )

    gt_nodes = tree_size(gt_tree)
    pred_nodes = tree_size(pred_tree)

    max_nodes = max(gt_nodes, pred_nodes)

    normalized_ted = (
        ted / max_nodes
        if max_nodes > 0
        else 0.0
    )

    return {
        "ted": float(ted),
        "normalized_ted": float(normalized_ted),
        "gt_nodes": gt_nodes,
        "pred_nodes": pred_nodes
    }


# ============================================================
# 10. 单个 JSON Pair
# ============================================================

def evaluate_pair(
    gt_path: str,
    pred_path: str
) -> Dict[str, Any]:

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    with open(pred_path, "r", encoding="utf-8") as f:
        pred_data = json.load(f)

    # -------------------------
    # Schema
    # -------------------------

    gt_schema = validate_schema(gt_data)
    pred_schema = validate_schema(pred_data)

    # -------------------------
    # TED
    # -------------------------

    ted_result = calculate_ted(
        gt_data,
        pred_data
    )

    return {

        "gt_schema_valid":
            int(gt_schema["valid"]),

        "pred_schema_valid":
            int(pred_schema["valid"]),

        "gt_schema_errors":
            gt_schema["num_errors"],

        "pred_schema_errors":
            pred_schema["num_errors"],

        "ted":
            ted_result["ted"],

        "normalized_ted":
            ted_result["normalized_ted"],

        "gt_nodes":
            ted_result["gt_nodes"],

        "pred_nodes":
            ted_result["pred_nodes"]
    }


# ============================================================
# 11. 批量评估
# ============================================================

def evaluate_directory(
    gt_dir: str,
    pred_dir: str,
    max_samples: int = 1000   # 新增参数
):

    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    results = []

    gt_files = sorted(gt_dir.glob("*.json"), key=lambda p: int(p.stem))

    # ================= 限定数量 =================
    if max_samples is not None and max_samples > 0:
        gt_files = gt_files[:max_samples]
    # ============================================

    print(f"共待评估 {len(gt_files)} 个文件")

    for gt_file in gt_files:

        pred_file = pred_dir / gt_file.name

        if not pred_file.exists():
            print(f"[WARNING] 找不到 Pred: {gt_file.name}")
            continue

        try:
            result = evaluate_pair(str(gt_file), str(pred_file))
            result["id"] = gt_file.stem
            results.append(result)

        except Exception as e:
            print(f"[ERROR] {gt_file.name}: {e}")

    return results



# ============================================================
# 12. 汇总
# ============================================================

def summarize(results):

    if not results:
        return {}

    schema_rate = sum(
        r["pred_schema_valid"]
        for r in results
    ) / len(results)

    mean_ted = sum(
        r["ted"]
        for r in results
    ) / len(results)

    mean_normalized_ted = sum(
        r["normalized_ted"]
        for r in results
    ) / len(results)

    return {

        "num_samples":
            len(results),

        "schema_validation_rate":
            schema_rate,

        "mean_ted":
            mean_ted,

        "mean_normalized_ted":
            mean_normalized_ted
    }


# ============================================================
# 13. Main
# ============================================================

if __name__ == "__main__":

    # 修改成你的路径
    GT_DIR = ""
    PRED_DIR = ""

    results = evaluate_directory(
        GT_DIR,
        PRED_DIR,
        max_samples=1000
    )

    summary = summarize(results)

    print("\n==============================")
    print("Evaluation Summary")
    print("==============================")

    for key, value in summary.items():
        print(f"{key}: {value}")