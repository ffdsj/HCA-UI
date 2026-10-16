#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# LGE修正
import os, re, json, copy, argparse
from glob import glob
from collections import defaultdict

SKIP_SYSTEM_CLASSES = {
    'com.android.internal.policy.PhoneWindow$DecorView',
    'android.view.ViewStub',
    'android.widget.Space',
}

def natural_key(path):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', os.path.basename(path))]

def get_root(data):
    if isinstance(data, dict):
        if 'activity' in data and isinstance(data['activity'], dict) and 'root' in data['activity']:
            return data['activity']['root']
        if 'root' in data:
            return data['root']
    return data

def valid_bounds(b):
    return isinstance(b, list) and len(b) == 4 and b[2] > b[0] and b[3] > b[1]

def area(b):
    return max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])

def overlaps(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    return iw > 0 and ih > 0

def contains(outer, inner):
    return outer[0] <= inner[0] and outer[1] <= inner[1] and outer[2] >= inner[2] and outer[3] >= inner[3]

def shifted_box(b, dx, dy):
    return [b[0]+dx, b[1]+dy, b[2]+dx, b[3]+dy]

class EvalView:
    def __init__(self, data):
        self.nodes = {}
        self.children = defaultdict(list)
        root = get_root(data)
        self.canvas = self._infer_canvas(root)
        self._walk(root, None, '1')

    def _infer_canvas(self, root):
        if isinstance(root, dict) and valid_bounds(root.get('bounds')):
            b = root['bounds']; return float(b[2]), float(b[3])
        mx, my = 1.0, 1.0
        def scan(n):
            nonlocal mx, my
            if not isinstance(n, dict): return
            b = n.get('bounds')
            if valid_bounds(b): mx, my = max(mx, float(b[2])), max(my, float(b[3]))
            for c in n.get('children', []) or []: scan(c)
        scan(root)
        return mx, my

    def _walk(self, node, parent_id, path):
        if not isinstance(node, dict): return
        b = node.get('bounds', [0,0,0,0])
        if not valid_bounds(b): return
        cls = node.get('class', '')
        skip = node.get('visibility', 'visible') in ('gone','invisible') or node.get('draw', True) is False or cls in SKIP_SYSTEM_CLASSES
        if skip:
            for i, c in enumerate(node.get('children', []) or []):
                self._walk(c, parent_id, f'{path}-s{i+1}')
            return
        nid = path
        self.nodes[nid] = {'parent_id': parent_id, 'ref': node}
        if parent_id is not None: self.children[parent_id].append(nid)
        for i, c in enumerate(node.get('children', []) or []):
            self._walk(c, nid, f'{path}-{i+1}')

    def bbox(self, nid):
        return self.nodes[nid]['ref']['bounds']

    def set_bbox(self, nid, b):
        old = self.nodes[nid]['ref']['bounds']
        if all(isinstance(v, int) for v in old):
            self.nodes[nid]['ref']['bounds'] = [int(round(v)) for v in b]
        else:
            self.nodes[nid]['ref']['bounds'] = [float(v) for v in b]

    def descendants(self, nid):
        out, stack = set(), [nid]
        while stack:
            cur = stack.pop()
            if cur in out: continue
            out.add(cur); stack.extend(self.children.get(cur, []))
        return out

    def union_box(self, ids):
        bs = [self.bbox(i) for i in ids if i in self.nodes]
        if not bs: return None
        return [min(b[0] for b in bs), min(b[1] for b in bs), max(b[2] for b in bs), max(b[3] for b in bs)]

    def shift_subtree(self, nid, dx, dy):
        for sid in self.descendants(nid):
            self.set_bbox(sid, shifted_box(self.bbox(sid), dx, dy))

def cvr_counts(view):
    checks = violations = containment = sibling = invalid = 0
    for nid, info in view.nodes.items():
        pid = info['parent_id']
        if pid is None or pid not in view.nodes: continue
        checks += 1
        if not contains(view.bbox(pid), view.bbox(nid)):
            violations += 1; containment += 1
    for pid, cids in view.children.items():
        for i in range(len(cids)):
            for j in range(i+1, len(cids)):
                checks += 1
                if overlaps(view.bbox(cids[i]), view.bbox(cids[j])):
                    violations += 1; sibling += 1
    for nid in view.nodes:
        checks += 1
        if area(view.bbox(nid)) <= 0:
            violations += 1; invalid += 1
    return {'checks':checks,'violations':violations,'containment':containment,'sibling':sibling,'invalid_area':invalid,'rate':violations/checks if checks else 0.0}

def sibling_conflict_count_for(view, nid, test_box=None):
    pid = view.nodes[nid]['parent_id']
    if pid is None: return 0
    b = test_box if test_box is not None else view.bbox(nid)
    return sum(1 for sid in view.children.get(pid, []) if sid != nid and overlaps(b, view.bbox(sid)))

def candidate_separation_shifts(moving, fixed):
    if not overlaps(moving, fixed): return []
    return [
        (fixed[0]-moving[2], 0.0),
        (fixed[2]-moving[0], 0.0),
        (0.0, fixed[1]-moving[3]),
        (0.0, fixed[3]-moving[1]),
    ]

def allowed_shift(dx, dy, canvas, max_shift_ratio):
    w, h = canvas
    return abs(dx) <= max_shift_ratio*max(w,1.0) and abs(dy) <= max_shift_ratio*max(h,1.0)

def best_sibling_move(view, moving_id, fixed_id, max_shift_ratio):
    mb, fb = view.bbox(moving_id), view.bbox(fixed_id)
    if not overlaps(mb, fb): return None
    subtree = view.descendants(moving_id)
    ub = view.union_box(subtree)
    pid = view.nodes[moving_id]['parent_id']
    parent_box = [0.0,0.0,view.canvas[0],view.canvas[1]] if pid is None or pid not in view.nodes else view.bbox(pid)
    old_conf = sibling_conflict_count_for(view, moving_id)
    best = None
    for dx, dy in candidate_separation_shifts(mb, fb):
        if not allowed_shift(dx, dy, view.canvas, max_shift_ratio): continue
        new_mb, new_ub = shifted_box(mb,dx,dy), shifted_box(ub,dx,dy)
        if not contains(parent_box, new_ub): continue
        if overlaps(new_mb, fb): continue
        new_conf = sibling_conflict_count_for(view, moving_id, new_mb)
        gain = old_conf - new_conf
        if gain <= 0: continue
        cost = abs(dx)/max(view.canvas[0],1.0) + abs(dy)/max(view.canvas[1],1.0)
        cand = (gain, cost, dx, dy)
        if best is None or gain > best[0] or (gain == best[0] and cost < best[1]): best = cand
    if best is None: return None
    return {'gain':best[0],'cost':best[1],'dx':best[2],'dy':best[3]}

def minimal_containment_shift(child, parent):
    cw, ch = child[2]-child[0], child[3]-child[1]
    pw, ph = parent[2]-parent[0], parent[3]-parent[1]
    if cw > pw or ch > ph: return None
    dx = parent[0]-child[0] if child[0] < parent[0] else (parent[2]-child[2] if child[2] > parent[2] else 0.0)
    dy = parent[1]-child[1] if child[1] < parent[1] else (parent[3]-child[3] if child[3] > parent[3] else 0.0)
    return dx, dy

def repair_containment(view, max_shift_ratio):
    changed = False
    def depth(nid):
        d, cur, seen = 0, nid, set()
        while cur in view.nodes:
            if cur in seen: break
            seen.add(cur); pid = view.nodes[cur]['parent_id']
            if pid is None: break
            d += 1; cur = pid
        return d
    for nid in sorted(view.nodes, key=depth):
        pid = view.nodes[nid]['parent_id']
        if pid is None or pid not in view.nodes: continue
        cb, pb = view.bbox(nid), view.bbox(pid)
        if contains(pb, cb): continue
        sh = minimal_containment_shift(cb, pb)
        if sh is None: continue
        dx, dy = sh
        if not allowed_shift(dx,dy,view.canvas,max_shift_ratio): continue
        ub = view.union_box(view.descendants(nid))
        if not contains(pb, shifted_box(ub,dx,dy)): continue
        if sibling_conflict_count_for(view,nid,shifted_box(cb,dx,dy)) > sibling_conflict_count_for(view,nid): continue
        view.shift_subtree(nid,dx,dy); changed = True
    return changed

def repair_siblings(view, max_iter, max_shift_ratio):
    for _ in range(max_iter):
        changed = False
        for pid, cids in list(view.children.items()):
            for i in range(len(cids)):
                for j in range(i+1, len(cids)):
                    a, b = cids[i], cids[j]
                    if not overlaps(view.bbox(a), view.bbox(b)): continue
                    ca = best_sibling_move(view,a,b,max_shift_ratio)
                    cb = best_sibling_move(view,b,a,max_shift_ratio)
                    candidates = [(a,ca)] if ca else []
                    if cb: candidates.append((b,cb))
                    if not candidates: continue
                    moving, best = sorted(candidates, key=lambda x:(-x[1]['gain'], x[1]['cost']))[0]
                    view.shift_subtree(moving,best['dx'],best['dy']); changed = True
        if not changed: break

def optimize_one(data, max_iter=5, max_shift_ratio=0.01):
    original = copy.deepcopy(data)
    work = copy.deepcopy(data)
    before = cvr_counts(EvalView(original))
    view = EvalView(work)
    repair_containment(view,max_shift_ratio)
    repair_siblings(view,max_iter,max_shift_ratio)
    repair_containment(view,max_shift_ratio)
    after = cvr_counts(view)
    # 只有同口径 CVR violation 真正下降才保留，否则整份回滚
    if after['checks'] == before['checks'] and after['violations'] < before['violations']:
        return work, before, after, True
    return original, before, before, False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--max_iter', type=int, default=5)
    ap.add_argument('--max_shift_ratio', type=float, default=0.01)
    ap.add_argument('--max_files', type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob(os.path.join(args.input_dir,'*.json')), key=natural_key)
    if args.max_files is not None: files = files[:args.max_files]
    accepted = unchanged = failed = 0
    bv = av = checks = 0
    for i, path in enumerate(files):
        try:
            with open(path,'r',encoding='utf-8') as f: data = json.load(f)
            out,before,after,changed = optimize_one(data,args.max_iter,args.max_shift_ratio)
            with open(os.path.join(args.output_dir,os.path.basename(path)),'w',encoding='utf-8') as f:
                json.dump(out,f,indent=2,ensure_ascii=False)
            bv += before['violations']; av += after['violations']; checks += before['checks']
            accepted += int(changed); unchanged += int(not changed)
            if (i+1)%100 == 0 or i+1 == len(files):
                print(f'[{i+1}/{len(files)}] accepted={accepted}, unchanged={unchanged}, failed={failed}')
        except Exception as e:
            failed += 1; print(f'[ERROR] {os.path.basename(path)}: {e}')
    print('\n'+'='*60)
    print(f'files={len(files)} accepted={accepted} unchanged={unchanged} failed={failed}')
    if checks:
        print(f'CVR before={bv/checks:.8f}')
        print(f'CVR after ={av/checks:.8f}')
        print(f'delta     ={(av-bv)/checks:+.8f}')

if __name__ == '__main__':
    main()
