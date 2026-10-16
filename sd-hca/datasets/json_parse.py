import re
import json
from typing import List, Dict, Optional, Tuple


# ---------------- 核心解析 ----------------
def parse_json_tree(data: dict) -> Tuple[List[dict], Dict[str, str]]:
    """
    兼容两种JSON格式：
    - 10.json: 直接是组件树
    - 100.json: 嵌套在 activity.root 下
    """
    # 🔥 新增：统一提取根节点
    if isinstance(data, dict):
        if 'activity' in data and isinstance(data['activity'], dict) and 'root' in data['activity']:
            # 100.json 格式
            root = data['activity']['root']
        elif 'root' in data:
            # 可能的中间格式
            root = data['root']
        else:
            # 10.json 格式（本身就是根）
            root = data
    else:
        raise ValueError("Invalid JSON format: root not found")
    
    nodes: List[dict] = []
    parent_map: Dict[str, str] = {}
    counter = [0]
    
    # 传入根节点开始DFS
    dfs(root, None, 0, "", nodes, parent_map, counter)
    
    return nodes, parent_map


def dfs(node, parent_id, depth, prefix, nodes, parent_map, counter):
    # 🔥 防御：跳过 None 节点（ Rico 数据偶尔会有脏数据）
    if node is None or not isinstance(node, dict):
        return ""

    # 只提取必要字段
    bounds = node.get('bounds', [0, 0, 0, 0])
    if len(bounds) != 4 or bounds[2] <= bounds[0]:
        return ""
    
    # 生成序列号（1, 1-1, 1-1-1...）
    if depth == 0:
        seq = "1"
    else:
        while len(counter) <= depth:
            counter.append(0)
        counter[depth] += 1
        seq = f"{prefix}-{counter[depth]}"
    
    # 只保留核心字段
    nodes.append({
        'ancestry_id': seq,
        'depth': depth,
        'parent_id': parent_id,
        'bounds': bounds,  # 仅用于空间定位生成mask
    })
    
    # 递归子节点
    for child in node.get('children', []):
        child_id = dfs(child, seq, depth + 1, seq, nodes, parent_map, counter)
        if child_id:
            parent_map[child_id] = seq
    
    return seq