# -*- coding: utf-8 -*-

import csv
import json
import math
from copy import deepcopy
from pathlib import Path


GENERATED_ROOT = Path(
    r""
)

ORIGINAL_ROOT = Path(
    r""
)

OUTPUT_ROOT = Path(
    r""
)


TEXT_MIN_IOU = 0.30

TYPE_MIN_IOU = 0.25

FALLBACK_MIN_IOU = 0.15
FALLBACK_CENTER_PX = 10.0
FALLBACK_SIZE_SIM = 0.50

OVERWRITE_EXISTING = False


# ============================================================
# 3. JSON 基础工具
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_text(value):
    if isinstance(value, str):
        return value.strip()

    # Rico 的 content-desc 常见为 [null] 或字符串数组
    if isinstance(value, list):
        parts = [
            str(x).strip()
            for x in value
            if x is not None and str(x).strip()
        ]
        return " ".join(parts)

    return ""


def extract_text(node: dict):
    """
    文字字段优先级：
      text > text-hint > content-desc
    """
    for key in ("text", "text-hint", "content-desc"):
        value = safe_text(node.get(key))
        if value:
            return value, key
    return "", None


def get_root(json_data):
    if (
        isinstance(json_data, dict)
        and isinstance(json_data.get("activity"), dict)
        and isinstance(json_data["activity"].get("root"), dict)
    ):
        return json_data["activity"]["root"]
    return json_data


def get_original_resolution(json_data):
    """
    与你推理代码的逻辑一致：
    优先 activity.root.bounds。
    """
    root = get_root(json_data)

    def valid_resolution(node):
        if not isinstance(node, dict):
            return None
        b = node.get("bounds")
        if isinstance(b, (list, tuple)) and len(b) == 4:
            x1, y1, x2, y2 = b
            try:
                w = float(x2)
                h = float(y2)
                if w > 0 and h > 0:
                    return w, h
            except Exception:
                pass
        return None

    res = valid_resolution(root)
    if res:
        return res

    for child in root.get("children", []) or []:
        res = valid_resolution(child)
        if res:
            return res

    return 1440.0, 2560.0


# ============================================================
# 4. 原始树展开与无效节点过滤
# ============================================================

def flatten_tree(node, path=()):
    """
    返回：
      [{"path": (...), "node": raw_node}, ...]
    """
    out = []

    if not isinstance(node, dict):
        return out

    out.append({
        "path": path,
        "node": node,
    })

    children = node.get("children", []) or []
    if isinstance(children, list):
        for i, child in enumerate(children):
            out.extend(
                flatten_tree(child, path + (i,))
            )

    return out


def valid_positive_bounds(bounds):
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 4:
        return False

    try:
        x1, y1, x2, y2 = map(float, bounds)
    except Exception:
        return False

    return x2 > x1 and y2 > y1


def raw_node_is_usable(node):
    """
    过滤明显无效节点：
    - bounds <= 0
    - visibility == gone/invisible
    - visible-to-user == false

    注意：
    visible-to-user 缺失时不直接删除。
    """
    if not isinstance(node, dict):
        return False

    bounds = node.get("bounds")
    if not valid_positive_bounds(bounds):
        return False

    visibility = str(
        node.get("visibility", "")
    ).strip().lower()

    if visibility in {"gone", "invisible"}:
        return False

    visible_to_user = node.get("visible-to-user", None)
    if visible_to_user is False:
        return False

    return True


def build_raw_candidates(original_json, canvas_w, canvas_h):
    """
    从原始 JSON 中提取可用于匹配的节点，
    并将原始 bounds 映射到 generated JSON 的 canvas 坐标系。
    """
    root = get_root(original_json)
    orig_w, orig_h = get_original_resolution(original_json)

    candidates = []

    for item in flatten_tree(root):
        path = item["path"]
        node = item["node"]

        if not raw_node_is_usable(node):
            continue

        x1, y1, x2, y2 = map(float, node["bounds"])

        scaled = {
            "x": x1 / orig_w * canvas_w,
            "y": y1 / orig_h * canvas_h,
            "width": (x2 - x1) / orig_w * canvas_w,
            "height": (y2 - y1) / orig_h * canvas_h,
        }

        text, text_key = extract_text(node)

        candidates.append({
            "raw_path": list(path),
            "raw_bounds": [x1, y1, x2, y2],
            "scaled_bounds": scaled,
            "text": text,
            "text_key": text_key,
            "class": node.get("class"),
            "resource_id": node.get("resource-id"),
            "pointer": node.get("pointer"),
            "clickable": node.get("clickable"),
            "raw_node": node,
        })

    return candidates


# ============================================================
# 5. 几何匹配
# ============================================================

def xyxy(bounds):
    """
    generated bounds 格式：
      {x, y, width, height}
    """
    x1 = float(bounds["x"])
    y1 = float(bounds["y"])
    x2 = x1 + float(bounds["width"])
    y2 = y1 + float(bounds["height"])
    return x1, y1, x2, y2


def bbox_iou(a, b):
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter

    if union <= 0:
        return 0.0

    return inter / union


def center_distance(a, b):
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)

    acx = (ax1 + ax2) / 2.0
    acy = (ay1 + ay2) / 2.0

    bcx = (bx1 + bx2) / 2.0
    bcy = (by1 + by2) / 2.0

    return math.hypot(acx - bcx, acy - bcy)


def size_similarity(a, b):
    """
    0~1。
    宽、高分别取 min/max 后相乘。
    """
    aw = max(float(a["width"]), 1e-6)
    ah = max(float(a["height"]), 1e-6)

    bw = max(float(b["width"]), 1e-6)
    bh = max(float(b["height"]), 1e-6)

    wr = min(aw, bw) / max(aw, bw)
    hr = min(ah, bh) / max(ah, bh)

    return wr * hr


def match_score(gen_bounds, src_bounds, canvas_w, canvas_h):
    iou = bbox_iou(gen_bounds, src_bounds)
    dist = center_distance(gen_bounds, src_bounds)
    size_sim = size_similarity(gen_bounds, src_bounds)

    diag = math.hypot(canvas_w, canvas_h)
    center_sim = max(
        0.0,
        1.0 - dist / max(diag * 0.10, 1.0)
    )

    score = (
        0.75 * iou
        + 0.15 * center_sim
        + 0.10 * size_sim
    )

    return {
        "score": score,
        "iou": iou,
        "center_distance": dist,
        "size_similarity": size_sim,
    }


def candidate_is_acceptable(metrics, min_iou):
    if metrics["iou"] >= min_iou:
        return True

    return (
        metrics["iou"] >= FALLBACK_MIN_IOU
        and metrics["center_distance"] <= FALLBACK_CENTER_PX
        and metrics["size_similarity"] >= FALLBACK_SIZE_SIM
    )


def greedy_one_to_one_match(
    components,
    candidates,
    canvas_w,
    canvas_h,
    min_iou,
):
    """
    对 component 与 candidate 做一对一贪心匹配。

    返回：
      {
        component_index: {
            "candidate_index": ...,
            "metrics": ...
        }
      }
    """
    pairs = []

    for ci, comp in enumerate(components):
        gb = comp.get("bounds")

        if not isinstance(gb, dict):
            continue

        required = {"x", "y", "width", "height"}
        if not required.issubset(gb):
            continue

        for ri, src in enumerate(candidates):
            metrics = match_score(
                gb,
                src["scaled_bounds"],
                canvas_w,
                canvas_h,
            )

            if candidate_is_acceptable(metrics, min_iou):
                pairs.append((
                    metrics["score"],
                    ci,
                    ri,
                    metrics,
                ))

    pairs.sort(key=lambda x: x[0], reverse=True)

    used_components = set()
    used_candidates = set()

    matches = {}

    for score, ci, ri, metrics in pairs:
        if ci in used_components:
            continue
        if ri in used_candidates:
            continue

        used_components.add(ci)
        used_candidates.add(ri)

        matches[ci] = {
            "candidate_index": ri,
            "metrics": metrics,
        }

    return matches


# ============================================================
# 6. Android class -> 输出 type
# ============================================================

def class_to_type(class_name):
    if not class_name:
        return None

    name = str(class_name).lower()

    if "edittext" in name:
        return "Input"

    if "radiobutton" in name:
        return "RadioButton"

    if "checkbox" in name:
        return "CheckBox"

    if (
        "button" in name
        and "radiobutton" not in name
        and "checkbox" not in name
    ):
        return "Button"

    if "textview" in name:
        return "Text"

    if "imageview" in name:
        return "Image"

    if "listview" in name or "recyclerview" in name:
        return "List"

    if "scrollview" in name:
        return "Container"

    container_keywords = (
        "layout",
        "viewgroup",
        "framelayout",
        "linearlayout",
        "relativelayout",
        "constraintlayout",
        "drawerlayout",
        "radiogroup",
    )

    if any(k in name for k in container_keywords):
        return "Container"

    return "View"


# ============================================================
# 7. 找原始 JSON
# ============================================================

def resolve_original_json(generated_json_path, generated_data):
    """
    优先：
    1. metadata.input_json
    2. ORIGINAL_ROOT / sample_id.json
    3. ORIGINAL_ROOT / generated文件名前缀.json
    """

    metadata = generated_data.get("metadata", {})

    input_json = metadata.get("input_json")
    if input_json:
        p = Path(input_json)
        if p.exists():
            return p

    sample_id = str(
        metadata.get("sample_id")
        or generated_json_path.stem.replace("_generated", "")
    )

    candidates = [
        ORIGINAL_ROOT / f"{sample_id}.json",
        ORIGINAL_ROOT / f"{sample_id}.JSON",
    ]

    for p in candidates:
        if p.exists():
            return p

    return None


# ============================================================
# 8. 单文件补全
# ============================================================

def enrich_one(generated_json_path: Path):
    generated = load_json(generated_json_path)

    canvas = generated.get("canvas", {})
    canvas_w = float(canvas.get("width", 512))
    canvas_h = float(canvas.get("height", 512))

    components = generated.get("components", [])

    original_json_path = resolve_original_json(
        generated_json_path,
        generated,
    )

    if original_json_path is None:
        return None, {
            "generated_json": str(generated_json_path),
            "status": "failed",
            "reason": "original_json_not_found",
        }, []

    original = load_json(original_json_path)

    raw_candidates = build_raw_candidates(
        original,
        canvas_w,
        canvas_h,
    )

    # --------------------------------------------------------
    # A. 类型候选：所有可用原始节点
    # --------------------------------------------------------

    type_candidates = [
        c
        for c in raw_candidates
        if c.get("class")
    ]

    type_matches = greedy_one_to_one_match(
        components=components,
        candidates=type_candidates,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        min_iou=TYPE_MIN_IOU,
    )

    # --------------------------------------------------------
    # B. 文字候选：仅非空 text / text-hint / content-desc
    # --------------------------------------------------------

    text_candidates = [
        c
        for c in raw_candidates
        if c.get("text")
    ]

    text_matches = greedy_one_to_one_match(
        components=components,
        candidates=text_candidates,
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        min_iou=TEXT_MIN_IOU,
    )

    enriched = deepcopy(generated)

    enriched_components = enriched.get("components", [])

    component_rows = []

    text_added = 0
    type_added = 0
    type_matched = 0
    text_matched = 0

    for ci, comp in enumerate(enriched_components):
        comp_id = comp.get("id")

        row = {
            "sample_id": enriched.get("metadata", {}).get("sample_id"),
            "component_id": comp_id,
            "generated_x": comp.get("bounds", {}).get("x"),
            "generated_y": comp.get("bounds", {}).get("y"),
            "generated_width": comp.get("bounds", {}).get("width"),
            "generated_height": comp.get("bounds", {}).get("height"),
            "old_type": comp.get("type"),
            "new_type": comp.get("type"),
            "old_text": comp.get("text", ""),
            "new_text": comp.get("text", ""),
            "type_match_iou": "",
            "text_match_iou": "",
            "source_class": "",
            "source_text": "",
            "source_bounds": "",
        }

        match_info = {}

        # ====================================================
        # Type
        # ====================================================

        if ci in type_matches:
            m = type_matches[ci]
            src = type_candidates[m["candidate_index"]]
            metrics = m["metrics"]

            type_matched += 1

            mapped_type = class_to_type(
                src.get("class")
            )

            old_type = str(comp.get("type", "") or "")
            raw_type = str(comp.get("raw_type", "") or "")

            should_fill_type = (
                OVERWRITE_EXISTING
                or old_type in {"", "View"}
                or raw_type.startswith("Component_")
            )

            if mapped_type and should_fill_type:
                comp["type"] = mapped_type
                comp["raw_type"] = src.get("class")
                type_added += 1

            match_info["type"] = {
                "method": "geometry",
                "iou": round(metrics["iou"], 6),
                "center_distance_px": round(
                    metrics["center_distance"], 3
                ),
                "size_similarity": round(
                    metrics["size_similarity"], 6
                ),
                "source_class": src.get("class"),
                "source_resource_id": src.get("resource_id"),
                "source_bounds": src.get("raw_bounds"),
                "source_path": src.get("raw_path"),
            }

            row["new_type"] = comp.get("type")
            row["type_match_iou"] = round(
                metrics["iou"], 6
            )
            row["source_class"] = src.get("class")
            row["source_bounds"] = json.dumps(
                src.get("raw_bounds"),
                ensure_ascii=False
            )

        # ====================================================
        # Text
        # ====================================================

        if ci in text_matches:
            m = text_matches[ci]
            src = text_candidates[m["candidate_index"]]
            metrics = m["metrics"]

            text_matched += 1

            old_text = safe_text(comp.get("text"))

            if OVERWRITE_EXISTING or not old_text:
                comp["text"] = src["text"]
                text_added += 1

            match_info["text"] = {
                "method": "geometry",
                "iou": round(metrics["iou"], 6),
                "center_distance_px": round(
                    metrics["center_distance"], 3
                ),
                "size_similarity": round(
                    metrics["size_similarity"], 6
                ),
                "source_text_key": src.get("text_key"),
                "source_text": src.get("text"),
                "source_class": src.get("class"),
                "source_resource_id": src.get("resource_id"),
                "source_bounds": src.get("raw_bounds"),
                "source_path": src.get("raw_path"),
            }

            row["new_text"] = comp.get("text", "")
            row["text_match_iou"] = round(
                metrics["iou"], 6
            )
            row["source_text"] = src.get("text")

            if not row["source_class"]:
                row["source_class"] = src.get("class")

            if not row["source_bounds"]:
                row["source_bounds"] = json.dumps(
                    src.get("raw_bounds"),
                    ensure_ascii=False
                )

        if match_info:
            comp["source_match"] = match_info

        component_rows.append(row)

    # --------------------------------------------------------
    # metadata：明确标注字段来源
    # --------------------------------------------------------

    metadata = enriched.setdefault("metadata", {})
    metadata["field_enrichment"] = {
        "performed": True,
        "model_rerun": False,
        "source": "original_input_json",
        "matching": "geometry_one_to_one",
        "note": (
            "text/type are recovered from the original input JSON "
            "by conservative geometric matching; they are not "
            "predicted by the diffusion model."
        ),
        "original_json": str(original_json_path),
        "text_min_iou": TEXT_MIN_IOU,
        "type_min_iou": TYPE_MIN_IOU,
    }

    sample_id = str(
        metadata.get("sample_id")
        or generated_json_path.stem.replace("_generated", "")
    )

    output_dir = OUTPUT_ROOT / sample_id
    output_path = (
        output_dir
        / f"{sample_id}_generated_enriched.json"
    )

    save_json(output_path, enriched)

    summary = {
        "sample_id": sample_id,
        "generated_json": str(generated_json_path),
        "original_json": str(original_json_path),
        "output_json": str(output_path),
        "status": "success",
        "component_count": len(components),
        "raw_usable_node_count": len(raw_candidates),
        "raw_text_node_count": len(text_candidates),
        "type_matched": type_matched,
        "type_added": type_added,
        "text_matched": text_matched,
        "text_added": text_added,
        "nonempty_text_after": sum(
            1
            for c in enriched_components
            if safe_text(c.get("text"))
        ),
    }

    return output_path, summary, component_rows


# ============================================================
# 9. 主程序
# ============================================================

def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    generated_files = sorted(
        GENERATED_ROOT.rglob("*_generated.json")
    )

    # 排除上一次补全结果
    generated_files = [
        p
        for p in generated_files
        if not p.name.endswith(
            "_generated_enriched.json"
        )
    ]

    if not generated_files:
        raise FileNotFoundError(
            f"在 {GENERATED_ROOT} 下没有找到 *_generated.json"
        )

    print("=" * 72)
    print(
        f"找到 generated JSON: "
        f"{len(generated_files)}"
    )
    print("=" * 72)

    summaries = []
    all_component_rows = []

    for i, path in enumerate(
        generated_files,
        start=1
    ):
        print(
            f"[{i:02d}/{len(generated_files):02d}] "
            f"{path}"
        )

        try:
            out_path, summary, rows = enrich_one(path)
            summaries.append(summary)
            all_component_rows.extend(rows)

            if summary["status"] == "success":
                print(
                    f"  text: "
                    f"{summary['text_added']} added, "
                    f"{summary['nonempty_text_after']} non-empty"
                )
                print(
                    f"  type: "
                    f"{summary['type_added']} added"
                )
                print(
                    f"  -> {out_path}"
                )
            else:
                print(
                    f"  FAILED: "
                    f"{summary.get('reason')}"
                )

        except Exception as e:
            print(
                f"  ERROR: {e}"
            )
            summaries.append({
                "generated_json": str(path),
                "status": "failed",
                "reason": str(e),
            })

    # ========================================================
    # 汇总 JSON
    # ========================================================

    success_rows = [
        x
        for x in summaries
        if x.get("status") == "success"
    ]

    global_summary = {
        "generated_file_count": len(generated_files),
        "success_file_count": len(success_rows),
        "failed_file_count": (
            len(generated_files)
            - len(success_rows)
        ),
        "total_text_added": sum(
            x.get("text_added", 0)
            for x in success_rows
        ),
        "total_type_added": sum(
            x.get("type_added", 0)
            for x in success_rows
        ),
        "files_with_at_least_one_text": sum(
            1
            for x in success_rows
            if x.get("nonempty_text_after", 0) > 0
        ),
        "per_file": summaries,
    }

    save_json(
        OUTPUT_ROOT / "enrichment_summary.json",
        global_summary,
    )

    # ========================================================
    # 每文件汇总 CSV
    # ========================================================

    summary_csv = OUTPUT_ROOT / "enrichment_summary.csv"

    summary_fields = [
        "sample_id",
        "status",
        "component_count",
        "raw_usable_node_count",
        "raw_text_node_count",
        "type_matched",
        "type_added",
        "text_matched",
        "text_added",
        "nonempty_text_after",
        "generated_json",
        "original_json",
        "output_json",
        "reason",
    ]

    with open(
        summary_csv,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary_fields,
            extrasaction="ignore",
        )
        writer.writeheader()

        for row in summaries:
            writer.writerow(row)

    # ========================================================
    # 逐组件匹配明细 CSV
    # ========================================================

    details_csv = OUTPUT_ROOT / "component_match_details.csv"

    detail_fields = [
        "sample_id",
        "component_id",
        "generated_x",
        "generated_y",
        "generated_width",
        "generated_height",
        "old_type",
        "new_type",
        "old_text",
        "new_text",
        "type_match_iou",
        "text_match_iou",
        "source_class",
        "source_text",
        "source_bounds",
    ]

    with open(
        details_csv,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=detail_fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(all_component_rows)

    print()
    print("=" * 72)
    print("字段补全完成")
    print(
        f"成功文件: "
        f"{global_summary['success_file_count']}"
        f"/{global_summary['generated_file_count']}"
    )
    print(
        f"至少补到一个文字节点的页面: "
        f"{global_summary['files_with_at_least_one_text']}"
        f"/{global_summary['success_file_count']}"
    )
    print(
        f"新增 text 总数: "
        f"{global_summary['total_text_added']}"
    )
    print(
        f"新增/修复 type 总数: "
        f"{global_summary['total_type_added']}"
    )
    print(
        f"汇总: "
        f"{OUTPUT_ROOT / 'enrichment_summary.json'}"
    )
    print(
        f"逐组件匹配: "
        f"{details_csv}"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()


    
# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import csv
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont


INPUT_ROOT = Path(r"")

OUTPUT_ROOT = Path(r"")

N_SAMPLES = 30
REQUIRE_EXACTLY_N = True

TEXT_REPLACEMENT = "确认"
WIDTH_SCALE = 1.10
BASE_MOVE_PIXELS = 16

CLEAN_OUTPUT_ROOT = True

DRAW_DEBUG_OUTLINE = False

SAVE_COMPARE_IMAGES = True

JSON_GLOBS = [
    "*_generated_enriched.json",
    "*_generated.json",
    "*.json",
]

EXCLUDED_JSON_NAMES = {
    "pairing_report.json",
    "generation_summary.json",
    "summary.json",
    "task_plan.json",
    "edit_log.json",
}


# ============================================================
# 2. 基础 I/O
# ============================================================

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def natural_key(value: str):
    parts = re.split(r"(\d+)", str(value))
    return [int(x) if x.isdigit() else x.lower() for x in parts]


def get_sample_id(path: Path, data: Optional[Dict[str, Any]] = None) -> str:
    if data:
        sid = data.get("metadata", {}).get("sample_id")
        if sid is not None and str(sid).strip():
            return str(sid)

    stem = path.stem
    if stem.endswith("_generated"):
        stem = stem[: -len("_generated")]
    return stem


def discover_generated_jsons(input_root: Path) -> List[Path]:
    """递归查找批量生成结果 JSON，并去重。"""
    found: Dict[str, Path] = {}

    # 优先补全后的 *_generated_enriched.json；其次原始 *_generated.json。
    # 这样 text/type 补全后无需改其他逻辑即可直接进入 E4。
    enriched = list(input_root.rglob("*_generated_enriched.json"))
    preferred = enriched if enriched else list(input_root.rglob("*_generated.json"))
    candidates = preferred

    if not candidates:
        candidates = list(input_root.rglob("*.json"))

    for p in candidates:
        if p.name in EXCLUDED_JSON_NAMES:
            continue
        if OUTPUT_ROOT in p.parents:
            continue
        if p.name.endswith("_gate_log.json"):
            continue

        try:
            data = read_json(p)
        except Exception:
            continue

        # 必须像模型输出 JSON：有 canvas + components
        if not isinstance(data.get("components"), list):
            continue
        if not isinstance(data.get("canvas"), dict):
            continue

        sid = get_sample_id(p, data)
        # 同一 sample_id 若出现多个 JSON：enriched > generated > 其他。
        old = found.get(sid)
        def rank(path: Path) -> int:
            if path.name.endswith("_generated_enriched.json"):
                return 3
            if path.name.endswith("_generated.json"):
                return 2
            return 1
        if old is None or rank(p) > rank(old):
            found[sid] = p

    paths = sorted(found.values(), key=lambda p: natural_key(get_sample_id(p)))
    return paths


# ============================================================
# 3. JSON 组件工具
# ============================================================

def comp_id(comp: Dict[str, Any]) -> Optional[int]:
    try:
        return int(comp.get("id"))
    except Exception:
        return None


def comp_map(data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    out = {}
    for c in data.get("components", []):
        cid = comp_id(c)
        if cid is not None:
            out[cid] = c
    return out


def children_map(data: Dict[str, Any]) -> Dict[int, List[int]]:
    cmap = comp_map(data)
    out: Dict[int, List[int]] = defaultdict(list)
    for cid, c in cmap.items():
        try:
            pid = c.get("parent_id")
            if pid is None:
                continue
            pid = int(pid)
        except Exception:
            continue
        if pid in cmap:
            out[pid].append(cid)
    return out


def descendants(data: Dict[str, Any], root_id: int) -> List[int]:
    ch = children_map(data)
    result = []
    stack = list(ch.get(root_id, []))
    seen = set()
    while stack:
        cid = stack.pop()
        if cid in seen:
            continue
        seen.add(cid)
        result.append(cid)
        stack.extend(ch.get(cid, []))
    return result


def get_bounds(comp: Dict[str, Any]) -> Optional[Dict[str, int]]:
    b = comp.get("bounds")
    if not isinstance(b, dict):
        return None
    try:
        x = int(round(float(b["x"])))
        y = int(round(float(b["y"])))
        w = int(round(float(b["width"])))
        h = int(round(float(b["height"])))
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return {"x": x, "y": y, "width": w, "height": h}


def set_bounds(comp: Dict[str, Any], bounds: Dict[str, int]) -> None:
    comp["bounds"] = {
        "x": int(bounds["x"]),
        "y": int(bounds["y"]),
        "width": int(bounds["width"]),
        "height": int(bounds["height"]),
    }


def get_text(comp: Dict[str, Any]) -> str:
    # 新版批量导出：text 直接在组件根部
    v = comp.get("text")
    if isinstance(v, str):
        return v

    # 兼容 content.text
    content = comp.get("content")
    if isinstance(content, dict):
        v = content.get("text")
        if isinstance(v, str):
            return v

    return ""


def set_text(comp: Dict[str, Any], value: str) -> None:
    if "text" in comp or not isinstance(comp.get("content"), dict):
        comp["text"] = value
    else:
        comp["content"]["text"] = value


def bbox_tuple(b: Dict[str, int]) -> Tuple[int, int, int, int]:
    return (b["x"], b["y"], b["x"] + b["width"], b["y"] + b["height"])


def contains(outer: Dict[str, int], inner: Dict[str, int]) -> bool:
    ox1, oy1, ox2, oy2 = bbox_tuple(outer)
    ix1, iy1, ix2, iy2 = bbox_tuple(inner)
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2


def overlaps(a: Dict[str, int], b: Dict[str, int]) -> bool:
    ax1, ay1, ax2, ay2 = bbox_tuple(a)
    bx1, by1, bx2, by2 = bbox_tuple(b)
    return max(ax1, bx1) < min(ax2, bx2) and max(ay1, by1) < min(ay2, by2)


def component_depths(data: Dict[str, Any]) -> Dict[int, int]:
    cmap = comp_map(data)
    memo: Dict[int, int] = {}

    def depth(cid: int, visiting=None) -> int:
        if cid in memo:
            return memo[cid]
        if visiting is None:
            visiting = set()
        if cid in visiting:
            return 0
        visiting.add(cid)
        c = cmap[cid]
        pid = c.get("parent_id")
        try:
            pid = int(pid) if pid is not None else None
        except Exception:
            pid = None
        if pid is None or pid not in cmap:
            d = 0
        else:
            d = depth(pid, visiting) + 1
        memo[cid] = d
        return d

    for cid in cmap:
        depth(cid)
    return memo


# ============================================================
# 4. JSON Renderer：只根据 JSON 重建图像
# ============================================================

def parse_hex_color(value: Any, default=(240, 240, 240)) -> Tuple[int, int, int]:
    if not isinstance(value, str):
        return default
    s = value.strip()
    if re.fullmatch(r"#[0-9a-fA-F]{6}", s):
        return tuple(int(s[i:i + 2], 16) for i in (1, 3, 5))
    return default


def choose_text_color(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    # 简单亮度判断
    r, g, b = bg
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    return (20, 20, 20) if lum > 165 else (245, 245, 245)


def load_font(size: int = 16):
    candidates = [
        Path(r"C:\Windows\Fonts\msyh.ttc"),     # 微软雅黑
        Path(r"C:\Windows\Fonts\msyhbd.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for p in candidates:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def render_json_image(data: Dict[str, Any]) -> Tuple[Image.Image, Dict[str, Any]]:
    """只根据 JSON 在内存中重建图像；供正式渲染和任务规划共同使用。"""
    canvas = data.get("canvas", {})
    try:
        width = int(canvas.get("width", 512))
        height = int(canvas.get("height", 512))
    except Exception:
        width, height = 512, 512

    if width <= 0 or height <= 0:
        raise ValueError(f"invalid canvas size: {width}x{height}")

    bg = parse_hex_color(canvas.get("backgroundColor"), (255, 255, 255))
    image = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(image)

    cmap = comp_map(data)
    depths = component_depths(data)

    # 父节点先画，子节点后画；同层大区域先画，小区域后画
    comps = []
    for cid, c in cmap.items():
        b = get_bounds(c)
        if b is None:
            continue
        area = b["width"] * b["height"]
        comps.append((depths.get(cid, 0), -area, cid, c, b))
    comps.sort(key=lambda x: (x[0], x[1], x[2]))

    rendered_count = 0
    text_count = 0

    for _, _, cid, c, b in comps:
        x1 = max(0, min(width, b["x"]))
        y1 = max(0, min(height, b["y"]))
        x2 = max(0, min(width, b["x"] + b["width"]))
        y2 = max(0, min(height, b["y"] + b["height"]))
        if x2 <= x1 or y2 <= y1:
            continue

        style = c.get("style", {}) if isinstance(c.get("style"), dict) else {}
        fill = parse_hex_color(
            style.get("backgroundColor") or style.get("backgroundColorRobust"),
            (235, 235, 235),
        )

        draw.rectangle([x1, y1, x2 - 1, y2 - 1], fill=fill)

        if DRAW_DEBUG_OUTLINE:
            draw.rectangle([x1, y1, x2 - 1, y2 - 1], outline=(30, 30, 30), width=1)

        text = get_text(c).strip()
        if text:
            font_size = max(10, min(22, int(max(12, (y2 - y1) * 0.38))))
            font = load_font(font_size)
            text_color = choose_text_color(fill)
            display_text = text[:40]
            try:
                tb = draw.textbbox((0, 0), display_text, font=font)
                tw = tb[2] - tb[0]
                th = tb[3] - tb[1]
            except Exception:
                tw, th = draw.textlength(display_text, font=font), font_size
            tx = x1 + max(2, ((x2 - x1) - int(tw)) // 2)
            ty = y1 + max(1, ((y2 - y1) - int(th)) // 2)
            draw.text((tx, ty), display_text, fill=text_color, font=font)
            text_count += 1

        rendered_count += 1

    info = {
        "ok": rendered_count > 0,
        "rendered_components": rendered_count,
        "rendered_text_components": text_count,
        "canvas": [width, height],
    }
    return image, info


def render_json(data: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    image, info = render_json_image(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    info["ok"] = bool(
        info["ok"] and out_path.exists() and out_path.stat().st_size > 0
    )
    return info


def image_change_metrics_from_images(a: Image.Image, b: Image.Image) -> Dict[str, float]:
    """用于任务规划：不落盘，直接判断候选编辑是否真的改变重渲染图。"""
    aa = np.asarray(a.convert("RGB"), dtype=np.int16)
    bb = np.asarray(b.convert("RGB"), dtype=np.int16)
    if aa.shape != bb.shape:
        return {"changed_pixel_ratio": 1.0, "mean_abs_diff": float("inf")}
    diff = np.abs(aa - bb)
    changed = np.any(diff > 0, axis=2)
    return {
        "changed_pixel_ratio": float(changed.mean()),
        "mean_abs_diff": float(diff.mean()),
    }


def make_compare(left_path: Path, right_path: Path, out_path: Path,
                 left_label="Before", right_label="After") -> None:
    a = Image.open(left_path).convert("RGB")
    b = Image.open(right_path).convert("RGB")
    h = max(a.height, b.height)
    label_h = 34
    gap = 16
    w = a.width + gap + b.width
    canvas = Image.new("RGB", (w, h + label_h), (250, 250, 250))
    canvas.paste(a, (0, label_h))
    canvas.paste(b, (a.width + gap, label_h))
    draw = ImageDraw.Draw(canvas)
    font = load_font(16)
    draw.text((8, 8), left_label, fill=(20, 20, 20), font=font)
    draw.text((a.width + gap + 8, 8), right_label, fill=(20, 20, 20), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def image_change_metrics(before_path: Path, after_path: Path) -> Dict[str, float]:
    a = np.asarray(Image.open(before_path).convert("RGB"), dtype=np.int16)
    b = np.asarray(Image.open(after_path).convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        return {"changed_pixel_ratio": 1.0, "mean_abs_diff": float("inf")}
    diff = np.abs(a - b)
    changed = np.any(diff > 0, axis=2)
    return {
        "changed_pixel_ratio": float(changed.mean()),
        "mean_abs_diff": float(diff.mean()),
    }


# ============================================================
# 5. 目标选择 / 任务计划
#    先计划，后执行，保证目标和目标值预先固定
# ============================================================

def candidate_text_target(data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    cmap = comp_map(data)
    ch = children_map(data)
    candidates = []

    for cid, c in cmap.items():
        b = get_bounds(c)
        if b is None:
            continue
        text = get_text(c).strip()
        ctype = str(c.get("type", "")).lower()

        # 优先已有真实文本；其次明确的文字类控件
        is_textual_type = any(k in ctype for k in ("text", "button", "input", "label"))
        if not text and not is_textual_type:
            continue
        if text == TEXT_REPLACEMENT:
            continue

        leaf_bonus = 1 if len(ch.get(cid, [])) == 0 else 0
        existing_text_bonus = 1 if text else 0
        area = b["width"] * b["height"]
        # 大而可见的叶子文字节点优先
        score = (existing_text_bonus, leaf_bonus, min(area, 100000), -cid)
        candidates.append((score, cid, c, b, text))

    if not candidates:
        return None, "no_text_node"

    candidates.sort(reverse=True, key=lambda x: x[0])
    _, cid, c, b, text = candidates[0]
    return {
        "target_id": cid,
        "target_ancestry_id": c.get("ancestry_id"),
        "before": {"text": text},
        "after": {"text": TEXT_REPLACEMENT},
        "allowed_changed_ids": [cid],
    }, ""


def candidate_move_leaf(data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    V2：根据父容器的真实剩余空间计算可行位置，而不是固定尝试 ±16 px。

    规则：
    1. 只选叶节点；
    2. 父节点必须存在，且当前叶节点已经位于父节点内部；
    3. 根据父容器与画布共同约束得到 x/y 的完整可行区间；
    4. 在区间边界、中点及当前坐标附近的可行位置中搜索；
    5. 优先不增加同父兄弟重叠；
    6. 预先用同一 JSON renderer 验证修改后确实有可见像素变化。
    """
    cmap = comp_map(data)
    child_map = children_map(data)

    canvas = data.get("canvas", {})
    cw = int(canvas.get("width", 512))
    chh = int(canvas.get("height", 512))
    canvas_b = {"x": 0, "y": 0, "width": cw, "height": chh}

    original_img, _ = render_json_image(data)

    candidates = []

    for cid, c in cmap.items():
        # 必须是叶节点
        if child_map.get(cid):
            continue

        b = get_bounds(c)
        if b is None:
            continue

        try:
            pid = int(c.get("parent_id"))
        except Exception:
            continue

        if pid not in cmap:
            continue

        pb = get_bounds(cmap[pid])
        if pb is None:
            continue

        if not contains(pb, b) or not contains(canvas_b, b):
            continue

        area = b["width"] * b["height"]
        if area < 16:
            continue

        # ----------------------------------------------------
        # 完整可行坐标区间，而不是固定步长
        # ----------------------------------------------------
        x_min = max(pb["x"], 0)
        x_max = min(
            pb["x"] + pb["width"] - b["width"],
            cw - b["width"],
        )
        y_min = max(pb["y"], 0)
        y_max = min(
            pb["y"] + pb["height"] - b["height"],
            chh - b["height"],
        )

        if x_min > x_max or y_min > y_max:
            continue

        # 若 x/y 都没有任何移动余量，则该叶节点不可移动
        if x_min == x_max == b["x"] and y_min == y_max == b["y"]:
            continue

        # 同父兄弟
        siblings = []
        for sid, sc in cmap.items():
            if sid == cid:
                continue
            try:
                spid = int(sc.get("parent_id"))
            except Exception:
                continue
            if spid == pid:
                sb = get_bounds(sc)
                if sb is not None:
                    siblings.append(sb)

        old_overlap_count = sum(overlaps(b, sb) for sb in siblings)

        # ----------------------------------------------------
        # 从真实可行区间构造确定性候选位置
        # ----------------------------------------------------
        def axis_positions(lo: int, hi: int, cur: int) -> List[int]:
            values = {
                int(lo),
                int(hi),
                int((lo + hi) // 2),
                int(round(lo + (hi - lo) * 0.25)),
                int(round(lo + (hi - lo) * 0.75)),
                int(cur),
            }
            # 再加入“利用剩余空间移动一半”的位置，仍由可行区间决定
            if cur > lo:
                values.add(int(round(cur - (cur - lo) / 2.0)))
            if cur < hi:
                values.add(int(round(cur + (hi - cur) / 2.0)))
            return sorted(v for v in values if lo <= v <= hi)

        xs = axis_positions(x_min, x_max, b["x"])
        ys = axis_positions(y_min, y_max, b["y"])

        position_candidates = []

        # 优先只改一个轴，再考虑二维移动，便于解释
        for x in xs:
            if x != b["x"]:
                position_candidates.append((x, b["y"]))
        for y in ys:
            if y != b["y"]:
                position_candidates.append((b["x"], y))
        for x in xs:
            for y in ys:
                if x == b["x"] and y == b["y"]:
                    continue
                position_candidates.append((x, y))

        # 去重并保持确定顺序
        seen = set()
        unique_positions = []
        for pos in position_candidates:
            if pos not in seen:
                seen.add(pos)
                unique_positions.append(pos)

        for nx, ny in unique_positions:
            nb = dict(b)
            nb["x"] = int(nx)
            nb["y"] = int(ny)

            if not contains(pb, nb) or not contains(canvas_b, nb):
                continue

            new_overlap_count = sum(overlaps(nb, sb) for sb in siblings)
            overlap_delta = max(0, new_overlap_count - old_overlap_count)

            # ------------------------------------------------
            # 用同一个 renderer 预检“真的能看见变化”
            # ------------------------------------------------
            probe = copy.deepcopy(data)
            probe_map = comp_map(probe)
            set_bounds(probe_map[cid], nb)
            probe_img, _ = render_json_image(probe)
            visual = image_change_metrics_from_images(original_img, probe_img)

            if visual["changed_pixel_ratio"] <= 0.0:
                continue

            displacement = math.hypot(
                nb["x"] - b["x"],
                nb["y"] - b["y"],
            )

            # 无新增冲突 > 可见变化更明显 > 位移更明显 > 节点面积较大
            score = (
                -overlap_delta,
                visual["changed_pixel_ratio"],
                displacement,
                area,
                -cid,
            )

            candidates.append({
                "score": score,
                "cid": cid,
                "pid": pid,
                "before": b,
                "after": nb,
                "overlap_delta": overlap_delta,
                "visual_ratio": visual["changed_pixel_ratio"],
                "displacement": displacement,
                "x_range": [x_min, x_max],
                "y_range": [y_min, y_max],
            })

    if not candidates:
        return None, "no_feasible_leaf_move"

    best = max(candidates, key=lambda x: x["score"])
    cid = best["cid"]

    return {
        "target_id": cid,
        "target_ancestry_id": cmap[cid].get("ancestry_id"),
        "parent_id": best["pid"],
        "before": {"bounds": best["before"]},
        "after": {"bounds": best["after"]},
        "allowed_changed_ids": [cid],
        "planning_note": (
            "dynamic_parent_free_space; "
            f"x_range={best['x_range']}; y_range={best['y_range']}; "
            f"displacement={best['displacement']:.2f}; "
            f"new_sibling_overlap_delta={best['overlap_delta']}; "
            f"predicted_changed_pixel_ratio={best['visual_ratio']:.6f}"
        ),
    }, ""


def candidate_resize_container(data: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    V2：容器宽度扩大 10% 时依次考虑三种几何方案：
      1) right      左边界固定，向右扩；
      2) left       右边界固定，向左扩；
      3) symmetric  左右两侧共同扩。

    每个候选必须：
    - 非根容器，且至少有一个输出子节点；
    - 扩大后仍在父容器和画布内；
    - 尽量不新增同父兄弟冲突；
    - 使用同一 JSON renderer 预检，确保最终图像确实出现可见变化。
    """
    cmap = comp_map(data)
    child_map = children_map(data)

    canvas = data.get("canvas", {})
    cw = int(canvas.get("width", 512))
    chh = int(canvas.get("height", 512))
    canvas_b = {"x": 0, "y": 0, "width": cw, "height": chh}

    original_img, _ = render_json_image(data)

    candidates = []

    for cid, child_ids in child_map.items():
        if not child_ids:
            continue

        c = cmap.get(cid)
        if c is None:
            continue

        try:
            pid = int(c.get("parent_id"))
        except Exception:
            continue

        # 非根容器必须有可定位的父容器
        if pid not in cmap:
            continue

        b = get_bounds(c)
        pb = get_bounds(cmap[pid])
        if b is None or pb is None:
            continue

        if not contains(pb, b) or not contains(canvas_b, b):
            continue

        new_w = max(
            b["width"] + 1,
            int(round(b["width"] * WIDTH_SCALE)),
        )
        delta = new_w - b["width"]

        if delta <= 0:
            continue

        # ----------------------------------------------------
        # 三种扩大策略
        # ----------------------------------------------------
        left_part = delta // 2

        raw_variants = [
            ("right", {
                "x": b["x"],
                "y": b["y"],
                "width": new_w,
                "height": b["height"],
            }),
            ("left", {
                "x": b["x"] - delta,
                "y": b["y"],
                "width": new_w,
                "height": b["height"],
            }),
            ("symmetric", {
                "x": b["x"] - left_part,
                "y": b["y"],
                "width": new_w,
                "height": b["height"],
            }),
        ]

        # 去掉坐标完全相同的重复方案
        variants = []
        seen_bounds = set()
        for strategy, nb in raw_variants:
            key = (nb["x"], nb["y"], nb["width"], nb["height"])
            if key in seen_bounds:
                continue
            seen_bounds.add(key)
            variants.append((strategy, nb))

        # 同父兄弟
        siblings = []
        for sid, sc in cmap.items():
            if sid == cid:
                continue
            try:
                spid = int(sc.get("parent_id"))
            except Exception:
                continue
            if spid == pid:
                sb = get_bounds(sc)
                if sb is not None:
                    siblings.append(sb)

        old_overlap_count = sum(overlaps(b, sb) for sb in siblings)
        area = b["width"] * b["height"]

        for strategy, nb in variants:
            if not contains(pb, nb) or not contains(canvas_b, nb):
                continue

            new_overlap_count = sum(overlaps(nb, sb) for sb in siblings)
            overlap_delta = max(0, new_overlap_count - old_overlap_count)

            # ------------------------------------------------
            # 预检：JSON 改了以后，renderer 的最终像素也必须真的变化
            # ------------------------------------------------
            probe = copy.deepcopy(data)
            probe_map = comp_map(probe)
            set_bounds(probe_map[cid], nb)
            probe_img, _ = render_json_image(probe)
            visual = image_change_metrics_from_images(original_img, probe_img)

            if visual["changed_pixel_ratio"] <= 0.0:
                # 例如扩大的区域最终完全被后绘制子节点覆盖，直接不选
                continue

            strategy_priority = {
                "right": 3,
                "left": 2,
                "symmetric": 1,
            }[strategy]

            # 无新增冲突 > 实际可见变化 > 大容器 > 策略稳定优先级
            score = (
                -overlap_delta,
                visual["changed_pixel_ratio"],
                area,
                strategy_priority,
                -cid,
            )

            candidates.append({
                "score": score,
                "cid": cid,
                "pid": pid,
                "before": b,
                "after": nb,
                "strategy": strategy,
                "overlap_delta": overlap_delta,
                "visual_ratio": visual["changed_pixel_ratio"],
            })

    if not candidates:
        return None, "no_feasible_visible_nonroot_container_resize"

    best = max(candidates, key=lambda x: x["score"])
    cid = best["cid"]
    allowed = [cid] + descendants(data, cid)

    return {
        "target_id": cid,
        "target_ancestry_id": cmap[cid].get("ancestry_id"),
        "parent_id": best["pid"],
        "before": {"bounds": best["before"]},
        "after": {"bounds": best["after"]},
        "allowed_changed_ids": allowed,
        "planning_note": (
            f"width_x{WIDTH_SCALE:.2f}; strategy={best['strategy']}; "
            f"new_sibling_overlap_delta={best['overlap_delta']}; "
            f"predicted_changed_pixel_ratio={best['visual_ratio']:.6f}"
        ),
    }, ""


def build_task_plan(sample_paths: List[Path]) -> List[Dict[str, Any]]:
    plan: List[Dict[str, Any]] = []

    for path in sample_paths:
        data = read_json(path)
        sid = get_sample_id(path, data)

        planners = [
            ("text_edit", candidate_text_target),
            ("move_leaf", candidate_move_leaf),
            ("resize_container", candidate_resize_container),
        ]

        for op_type, fn in planners:
            item, reason = fn(data)
            record = {
                "sample_id": sid,
                "source_json": str(path.resolve()),
                "operation": op_type,
                "plan_status": "planned" if item else "unavailable",
                "failure_reason": reason if not item else None,
            }
            if item:
                record.update(item)
            plan.append(record)

    return plan


# ============================================================
# 6. 编辑与验收
# ============================================================

def apply_planned_edit(original: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
    edited = copy.deepcopy(original)
    cmap = comp_map(edited)
    tid = int(task["target_id"])
    if tid not in cmap:
        raise KeyError(f"target component {tid} not found")
    c = cmap[tid]

    op = task["operation"]
    if op == "text_edit":
        set_text(c, task["after"]["text"])
    elif op == "move_leaf":
        set_bounds(c, task["after"]["bounds"])
    elif op == "resize_container":
        set_bounds(c, task["after"]["bounds"])
    else:
        raise ValueError(f"unknown operation: {op}")

    return edited


def changed_component_ids(before: Dict[str, Any], after: Dict[str, Any]) -> List[int]:
    bm = comp_map(before)
    am = comp_map(after)
    ids = sorted(set(bm) | set(am))
    changed = []
    for cid in ids:
        if bm.get(cid) != am.get(cid):
            changed.append(cid)
    return changed


def verify_edit(original: Dict[str, Any], edited: Dict[str, Any], task: Dict[str, Any]) -> Tuple[bool, str]:
    tid = int(task["target_id"])
    om = comp_map(original)
    em = comp_map(edited)
    if tid not in om or tid not in em:
        return False, "target_missing_after_edit"

    op = task["operation"]
    if op == "text_edit":
        if get_text(em[tid]) != task["after"]["text"]:
            return False, "text_not_updated"
    elif op in ("move_leaf", "resize_container"):
        if get_bounds(em[tid]) != task["after"]["bounds"]:
            return False, "bounds_not_updated"

    # 层级关系必须完全不变
    if original.get("relationships") != edited.get("relationships"):
        return False, "relationships_changed"

    return True, ""


def safe_folder_name(value: str) -> str:
    s = re.sub(r"[^0-9A-Za-z._-]+", "_", str(value))
    return s or "sample"


def execute_task(task: Dict[str, Any], sample_dir: Path,
                 original: Dict[str, Any], original_render_path: Path) -> Dict[str, Any]:
    op = task["operation"]
    op_idx = {"text_edit": "01", "move_leaf": "02", "resize_container": "03"}[op]
    op_dir = sample_dir / f"{op_idx}_{op}"
    op_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "sample_id": task["sample_id"],
        "operation": op,
        "plan_status": task["plan_status"],
        "target_id": task.get("target_id"),
        "target_ancestry_id": task.get("target_ancestry_id"),
        "parent_id": task.get("parent_id"),
        "before": task.get("before"),
        "requested_after": task.get("after"),
        "planning_note": task.get("planning_note"),
        "json_edit_success": False,
        "render_success": False,
        "visual_reflected": False,
        "edit_and_render_success": False,
        "changed_component_ids": [],
        "out_of_scope_changed_ids": [],
        "out_of_scope_change_count": 0,
        "changed_pixel_ratio": 0.0,
        "mean_abs_diff": 0.0,
        "failure_reason": task.get("failure_reason"),
        "edited_json": None,
        "rendered_image": None,
        "compare_image": None,
    }

    if task["plan_status"] != "planned":
        return log

    try:
        edited = apply_planned_edit(original, task)
        ok, reason = verify_edit(original, edited, task)
        log["json_edit_success"] = bool(ok)
        if not ok:
            log["failure_reason"] = reason
            return log

        edited_json_path = op_dir / "edited.json"
        render_path = op_dir / "render.png"
        compare_path = op_dir / "compare.png"

        write_json(edited_json_path, edited)
        log["edited_json"] = str(edited_json_path.resolve())

        render_info = render_json(edited, render_path)
        log["render_success"] = bool(render_info["ok"])
        log["rendered_components"] = render_info["rendered_components"]
        log["rendered_text_components"] = render_info["rendered_text_components"]
        log["rendered_image"] = str(render_path.resolve())

        changed_ids = changed_component_ids(original, edited)
        log["changed_component_ids"] = changed_ids
        allowed = set(int(x) for x in task.get("allowed_changed_ids", []))
        out_scope = [cid for cid in changed_ids if cid not in allowed]
        log["out_of_scope_changed_ids"] = out_scope
        log["out_of_scope_change_count"] = len(out_scope)

        metrics = image_change_metrics(original_render_path, render_path)
        log.update(metrics)
        log["visual_reflected"] = metrics["changed_pixel_ratio"] > 0.0

        if SAVE_COMPARE_IMAGES:
            make_compare(original_render_path, render_path, compare_path,
                         left_label="Original JSON render", right_label=f"After {op}")
            log["compare_image"] = str(compare_path.resolve())

        log["edit_and_render_success"] = bool(
            log["json_edit_success"]
            and log["render_success"]
            and log["visual_reflected"]
            and log["out_of_scope_change_count"] == 0
        )

        if not log["edit_and_render_success"] and not log["failure_reason"]:
            if not log["render_success"]:
                log["failure_reason"] = "render_failed"
            elif not log["visual_reflected"]:
                log["failure_reason"] = "render_has_no_visible_change"
            elif log["out_of_scope_change_count"] > 0:
                log["failure_reason"] = "out_of_scope_json_changes"

    except Exception as e:
        log["failure_reason"] = f"exception: {type(e).__name__}: {e}"

    return log


# ============================================================
# 7. 汇总
# ============================================================

def write_operation_csv(logs: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "sample_id", "operation", "plan_status", "target_id", "target_ancestry_id", "parent_id",
        "json_edit_success", "render_success", "visual_reflected", "edit_and_render_success",
        "changed_pixel_ratio", "mean_abs_diff", "out_of_scope_change_count",
        "failure_reason", "edited_json", "rendered_image", "compare_image",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in logs:
            writer.writerow(row)


def build_summary(logs: List[Dict[str, Any]], sample_count: int) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "sample_count": sample_count,
        "operations_per_sample": 3,
        "total_operations": sample_count * 3,
        "operations": {},
    }

    for op in ("text_edit", "move_leaf", "resize_container"):
        rows = [r for r in logs if r["operation"] == op]
        failures = Counter(r.get("failure_reason") or "" for r in rows if not r["edit_and_render_success"])
        failures.pop("", None)
        summary["operations"][op] = {
            "denominator": sample_count,
            "planned": sum(r["plan_status"] == "planned" for r in rows),
            "json_edit_success": sum(bool(r["json_edit_success"]) for r in rows),
            "render_success": sum(bool(r["render_success"]) for r in rows),
            "visual_reflected": sum(bool(r["visual_reflected"]) for r in rows),
            "edit_and_render_success": sum(bool(r["edit_and_render_success"]) for r in rows),
            "success_rate": (
                sum(bool(r["edit_and_render_success"]) for r in rows) / sample_count
                if sample_count else 0.0
            ),
            "out_of_scope_change_total": sum(int(r["out_of_scope_change_count"]) for r in rows),
            "failure_reasons": dict(failures),
        }

    summary["overall"] = {
        "edit_and_render_success": sum(bool(r["edit_and_render_success"]) for r in logs),
        "denominator": sample_count * 3,
        "success_rate": (
            sum(bool(r["edit_and_render_success"]) for r in logs) / (sample_count * 3)
            if sample_count else 0.0
        ),
        "out_of_scope_change_total": sum(int(r["out_of_scope_change_count"]) for r in logs),
    }
    return summary


# ============================================================
# 8. 主程序
# ============================================================

def main():
    print("=" * 78)
    print("E4 batch V2: output JSON -> edit JSON -> JSON renderer -> image")
    print("=" * 78)

    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"INPUT_ROOT 不存在: {INPUT_ROOT}")

    paths = discover_generated_jsons(INPUT_ROOT)
    print(f"发现模型输出 JSON: {len(paths)}")

    if REQUIRE_EXACTLY_N and len(paths) != N_SAMPLES:
        raise RuntimeError(
            f"要求正好 {N_SAMPLES} 个结果，但找到 {len(paths)} 个。\n"
            "请把 INPUT_ROOT 指向这 30 个批量结果所在目录；本程序不会自动换样本。"
        )

    if len(paths) < N_SAMPLES:
        raise RuntimeError(f"结果不足 {N_SAMPLES} 个，当前只有 {len(paths)} 个")

    # 如果目录里多于 30 个，但用户明确要这批 30 个，可改 REQUIRE_EXACTLY_N=False，
    # 程序按 sample_id 自然排序后取前 30 个。
    paths = paths[:N_SAMPLES]

    print("处理样本：")
    for p in paths:
        try:
            d = read_json(p)
            sid = get_sample_id(p, d)
        except Exception:
            sid = p.stem
        print(f"  {sid}: {p}")

    if CLEAN_OUTPUT_ROOT and OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # A. 先生成完整任务计划，再执行任何修改
    # --------------------------------------------------------
    plan = build_task_plan(paths)
    plan_path = OUTPUT_ROOT / "task_plan.jsonl"
    plan_path.write_text("", encoding="utf-8")
    for row in plan:
        append_jsonl(plan_path, row)

    print(f"\n任务计划已固定: {plan_path}")
    print(f"计划操作总数: {len(plan)} (= {N_SAMPLES} × 3)")

    # 按 sample_id 组织任务
    tasks_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in plan:
        tasks_by_sample[str(row["sample_id"])].append(row)

    # --------------------------------------------------------
    # B. 执行 90 次操作
    # --------------------------------------------------------
    op_log_path = OUTPUT_ROOT / "operation_log.jsonl"
    op_log_path.write_text("", encoding="utf-8")
    all_logs: List[Dict[str, Any]] = []

    for index, source_path in enumerate(paths, start=1):
        original = read_json(source_path)
        sid = get_sample_id(source_path, original)
        sample_dir = OUTPUT_ROOT / "samples" / safe_folder_name(sid)
        sample_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[{index:02d}/{N_SAMPLES}] sample={sid}")

        # 保存这一页用于 E4 的原始输出 JSON
        original_json_copy = sample_dir / "00_original.json"
        write_json(original_json_copy, original)

        # 原始 JSON 渲染图（同一 renderer）
        original_render_path = sample_dir / "00_original_render.png"
        original_render_info = render_json(original, original_render_path)
        if not original_render_info["ok"]:
            print("  [WARN] 原始 JSON 渲染失败，该页三个操作都会记录相应失败")

        # 固定顺序执行，但每个任务都从 original deepcopy 开始，互不叠加
        order = {"text_edit": 1, "move_leaf": 2, "resize_container": 3}
        tasks = sorted(tasks_by_sample[sid], key=lambda x: order[x["operation"]])

        for task in tasks:
            log = execute_task(task, sample_dir, original, original_render_path)
            log["source_json"] = str(source_path.resolve())
            log["original_json_copy"] = str(original_json_copy.resolve())
            log["original_render"] = str(original_render_path.resolve())
            all_logs.append(log)
            append_jsonl(op_log_path, log)

            status = "PASS" if log["edit_and_render_success"] else "FAIL"
            print(
                f"  {task['operation']:<16} {status}"
                + (f" | target={task.get('target_id')}" if task.get("target_id") is not None else "")
                + (f" | {log.get('failure_reason')}" if log.get("failure_reason") else "")
            )

    # --------------------------------------------------------
    # C. 汇总
    # --------------------------------------------------------
    csv_path = OUTPUT_ROOT / "operation_log.csv"
    write_operation_csv(all_logs, csv_path)

    summary = build_summary(all_logs, N_SAMPLES)
    summary_path = OUTPUT_ROOT / "summary.json"
    write_json(summary_path, summary)

    print("\n" + "=" * 78)
    print("E4 批量实验完成")
    print("=" * 78)
    for op, s in summary["operations"].items():
        print(
            f"{op:<18}: {s['edit_and_render_success']:>2}/{s['denominator']} "
            f"({s['success_rate'] * 100:.1f}%), "
            f"out-of-scope changes={s['out_of_scope_change_total']}"
        )
    o = summary["overall"]
    print(
        f"OVERALL            : {o['edit_and_render_success']}/{o['denominator']} "
        f"({o['success_rate'] * 100:.1f}%)"
    )
    print(f"task plan : {plan_path}")
    print(f"JSONL log : {op_log_path}")
    print(f"CSV log   : {csv_path}")
    print(f"summary   : {summary_path}")
    print(f"images    : {OUTPUT_ROOT / 'samples'}")


if __name__ == "__main__":
    main()
