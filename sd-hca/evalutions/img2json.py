import csv
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq


# ============================================================
# 1. Configuration
# ============================================================

MODEL_PATH = ""

# 改成你的截图目录
INPUT_DIR = r""

# Rico 风格 JSON 输出目录
OUTPUT_JSON_DIR = r""

# 保存 ScreenVLM 原始 ScreenTag，强烈建议保留
OUTPUT_RAW_DIR = r""

# 每张图的解析日志
LOG_CSV = r""

# 先跑 50 张检查；确认后改成 None 跑完整 6400
MAX_IMAGES: Optional[int] = 6400

# 从数值排序后的第几张开始
START_INDEX = 0

# 已存在结果是否跳过，便于中断后继续
SKIP_EXISTING = True

# ScreenVLM 官方提示词
PROMPT_TEXT = "Generate the screen representation for this UI:"

# ScreenVLM loc token 的归一化范围
NORM_SIZE = 500

# 官方示例使用 6192
MAX_NEW_TOKENS = 6192

# 固定 greedy decoding，避免随机采样
DO_SAMPLE = False

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ============================================================
# 2. Utilities
# ============================================================

def natural_file_key(filename: str):
    """
    数字文件名按数值排序：
      0.png, 1.png, 2.png, ..., 10.png
    其他文件名排在后面。
    """
    stem = Path(filename).stem
    if stem.isdigit():
        return (0, int(stem), "")
    nums = re.findall(r"\d+", stem)
    if nums:
        return (1, tuple(int(x) for x in nums), stem.lower())
    return (2, (), stem.lower())


def ensure_dirs():
    os.makedirs(OUTPUT_JSON_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RAW_DIR, exist_ok=True)
    log_parent = os.path.dirname(LOG_CSV)
    if log_parent:
        os.makedirs(log_parent, exist_ok=True)


def select_device_and_dtype() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        device = "cuda"
        # bf16 支持良好时优先 bf16，否则 fp16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        device = "cpu"
        dtype = torch.float32
    return device, dtype


# ============================================================
# 3. ScreenVLM inference
# ============================================================

def load_screenvlm():
    device, dtype = select_device_and_dtype()

    print(f"[INFO] device={device}, dtype={dtype}")
    print(f"[INFO] loading processor: {MODEL_PATH}")
    processor = AutoProcessor.from_pretrained(MODEL_PATH,local_files_only=True)

    print(f"[INFO] loading model: {MODEL_PATH}")
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=dtype,
        local_files_only=True
    ).to(device)

    model.eval()
    return processor, model, device


@torch.inference_mode()
def infer_one(
    image: Image.Image,
    processor,
    model,
    device: str,
) -> str:
    """
    单张截图 -> ScreenTag 原始文本。
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_TEXT},
            ],
        },
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=prompt,
        images=[image],
        return_tensors="pt",
    ).to(device)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=DO_SAMPLE,
    )

    prompt_length = inputs["input_ids"].shape[1]

    output = processor.batch_decode(
        generated_ids[:, prompt_length:],
        skip_special_tokens=False,
    )[0].lstrip()

    return output


# ============================================================
# 4. Parse nested ScreenTag
# ============================================================

TOKEN_RE = re.compile(
    r"</?[A-Za-z][A-Za-z0-9_-]*>|<loc_\d+>"
)

LOC_RE = re.compile(r"<loc_(\d+)>")


def crop_to_screentag(text: str) -> str:
    """
    去掉模型输出中 ScreenTag 前后的其它特殊 token / 文字。
    """
    start = text.find("<screentag>")
    end = text.rfind("</screentag>")

    if start >= 0 and end >= 0 and end > start:
        end += len("</screentag>")
        return text[start:end]

    if start >= 0:
        return text[start:]

    return text


def parse_nested_screentag(text: str) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    将 ScreenVLM 的嵌套 ScreenTag 转成中间树。

    返回：
      top_nodes: 顶层 UI 节点
      stats:
        raw_nodes
        unmatched_closing_tags
        unclosed_nodes

    中间节点：
    {
        "tag": "navigation_bar",
        "locs": [0, 0, 500, 30],
        "text_chunks": [],
        "children": [...]
    }

    注意：
    官方 README 里的正则示例会把元素扁平化；
    本函数使用 stack 保留 ScreenTag 的原生嵌套层级。
    """
    text = crop_to_screentag(text)

    top_nodes: List[Dict[str, Any]] = []
    stack: List[Dict[str, Any]] = []

    stats = {
        "raw_nodes": 0,
        "unmatched_closing_tags": 0,
        "unclosed_nodes": 0,
    }

    cursor = 0

    for match in TOKEN_RE.finditer(text):
        # token 之前的普通文本属于当前最内层节点
        between = text[cursor:match.start()]
        if stack and between:
            stack[-1]["text_chunks"].append(between)

        token = match.group(0)
        cursor = match.end()

        # loc token
        loc_m = LOC_RE.fullmatch(token)
        if loc_m:
            if stack:
                stack[-1]["locs"].append(int(loc_m.group(1)))
            continue

        # closing tag
        if token.startswith("</"):
            tag = token[2:-1]

            if tag == "screentag":
                continue

            # 正常情况下应关闭栈顶。
            # 如果模型生成轻微错配，则向上寻找同名 opening tag，尽量保留可解析部分。
            found_idx = None
            for idx in range(len(stack) - 1, -1, -1):
                if stack[idx]["tag"] == tag:
                    found_idx = idx
                    break

            if found_idx is None:
                stats["unmatched_closing_tags"] += 1
                continue

            # 被跨过的节点视为未正常闭合，但层级仍保留
            if found_idx != len(stack) - 1:
                stats["unclosed_nodes"] += len(stack) - 1 - found_idx

            stack = stack[:found_idx]
            continue

        # opening tag
        tag = token[1:-1]

        if tag == "screentag":
            continue

        node = {
            "tag": tag,
            "locs": [],
            "text_chunks": [],
            "children": [],
        }

        stats["raw_nodes"] += 1

        if stack:
            stack[-1]["children"].append(node)
        else:
            top_nodes.append(node)

        stack.append(node)

    # 尾部普通文本
    if stack and cursor < len(text):
        stack[-1]["text_chunks"].append(text[cursor:])

    if stack:
        stats["unclosed_nodes"] += len(stack)

    return top_nodes, stats


# ============================================================
# 5. ScreenTag tree -> Rico-style tree
# ============================================================

def locs_to_bbox(
    locs: List[int],
    width: int,
    height: int,
) -> Optional[List[int]]:
    """
    ScreenVLM:
      <left><top><right><bottom>, each in [0, 500]

    转成原图像素坐标 [x1, y1, x2, y2]。
    """
    if len(locs) < 4:
        return None

    l, t, r, b = locs[:4]

    l = max(0, min(int(l), NORM_SIZE))
    t = max(0, min(int(t), NORM_SIZE))
    r = max(0, min(int(r), NORM_SIZE))
    b = max(0, min(int(b), NORM_SIZE))

    if r < l:
        l, r = r, l
    if b < t:
        t, b = b, t

    x1 = int(round(l / NORM_SIZE * width))
    y1 = int(round(t / NORM_SIZE * height))
    x2 = int(round(r / NORM_SIZE * width))
    y2 = int(round(b / NORM_SIZE * height))

    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return [x1, y1, x2, y2]


def clean_direct_text(text_chunks: List[str]) -> Optional[str]:
    text = " ".join(x.strip() for x in text_chunks if x.strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def convert_node_or_promote(
    node: Dict[str, Any],
    width: int,
    height: int,
    stats: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    将一个 ScreenTag 节点转换成 Rico 节点。

    若某节点 bbox 缺失/非法：
      - 不人工猜 bbox；
      - 删除该无效节点本身；
      - 将其有效 children 提升到上一层。
    这样不会用 GT，也不会额外用 bbox 包含关系重建 hierarchy。
    """
    converted_children: List[Dict[str, Any]] = []

    for child in node.get("children", []):
        converted_children.extend(
            convert_node_or_promote(
                child,
                width,
                height,
                stats,
            )
        )

    bbox = locs_to_bbox(
        node.get("locs", []),
        width,
        height,
    )

    if bbox is None:
        stats["invalid_bbox_nodes"] += 1
        stats["promoted_children"] += len(converted_children)
        return converted_children

    direct_text = clean_direct_text(
        node.get("text_chunks", [])
    )

    rico_node: Dict[str, Any] = {
        "bounds": bbox,
        # 保留 ScreenVLM 原标签；当前结构-only TED 不依赖该标签
        "class": str(node.get("tag", "View")),
        "children": converted_children,
    }

    if direct_text is not None:
        rico_node["text"] = direct_text

    stats["valid_nodes"] += 1

    return [rico_node]


def screentag_to_rico(
    raw_text: str,
    width: int,
    height: int,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    top_nodes, parse_stats = parse_nested_screentag(raw_text)

    convert_stats = {
        "valid_nodes": 0,
        "invalid_bbox_nodes": 0,
        "promoted_children": 0,
    }

    children: List[Dict[str, Any]] = []

    for node in top_nodes:
        children.extend(
            convert_node_or_promote(
                node,
                width,
                height,
                convert_stats,
            )
        )

    rico = {
        "activity": {
            "root": {
                "bounds": [0, 0, width, height],
                "class": "Root",
                "children": children,
            }
        }
    }

    stats = {
        **parse_stats,
        **convert_stats,
        "top_level_nodes": len(children),
    }

    return rico, stats


# ============================================================
# 6. Single image
# ============================================================

def process_one(
    image_path: str,
    json_path: str,
    raw_path: str,
    processor,
    model,
    device: str,
) -> Dict[str, Any]:

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    raw_text = infer_one(
        image,
        processor,
        model,
        device,
    )

    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    rico_json, stats = screentag_to_rico(
        raw_text,
        width,
        height,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            rico_json,
            f,
            indent=2,
            ensure_ascii=False,
        )

    return {
        "width": width,
        "height": height,
        **stats,
    }


# ============================================================
# 7. Batch
# ============================================================

LOG_FIELDS = [
    "filename",
    "status",
    "width",
    "height",
    "raw_nodes",
    "valid_nodes",
    "invalid_bbox_nodes",
    "promoted_children",
    "top_level_nodes",
    "unmatched_closing_tags",
    "unclosed_nodes",
    "json_path",
    "raw_path",
    "error",
]


def append_log(row: Dict[str, Any]):
    exists = os.path.exists(LOG_CSV)

    with open(
        LOG_CSV,
        "a",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=LOG_FIELDS,
        )

        if not exists:
            writer.writeheader()

        writer.writerow({
            k: row.get(k, "")
            for k in LOG_FIELDS
        })


def main():
    ensure_dirs()

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(IMAGE_EXTS)
    ]
    files.sort(key=natural_file_key)

    files = files[START_INDEX:]

    if MAX_IMAGES is not None:
        files = files[:MAX_IMAGES]

    print("=" * 70)
    print("ScreenVLM -> nested ScreenTag -> Rico-style JSON")
    print("=" * 70)
    print(f"INPUT_DIR      : {INPUT_DIR}")
    print(f"OUTPUT_JSON_DIR: {OUTPUT_JSON_DIR}")
    print(f"OUTPUT_RAW_DIR : {OUTPUT_RAW_DIR}")
    print(f"START_INDEX    : {START_INDEX}")
    print(f"MAX_IMAGES     : {MAX_IMAGES}")
    print(f"Selected       : {len(files)} images")
    print("=" * 70)

    if not files:
        print("[ERROR] No images found.")
        return

    processor, model, device = load_screenvlm()

    success = 0
    failed = 0
    valid_node_counts: List[int] = []

    for fname in tqdm(files, desc="ScreenVLM parsing"):
        stem = Path(fname).stem

        image_path = os.path.join(INPUT_DIR, fname)
        json_path = os.path.join(
            OUTPUT_JSON_DIR,
            f"{stem}.json",
        )
        raw_path = os.path.join(
            OUTPUT_RAW_DIR,
            f"{stem}.txt",
        )

        # 中断后继续
        if (
            SKIP_EXISTING
            and os.path.exists(json_path)
            and os.path.exists(raw_path)
        ):
            continue

        try:
            stats = process_one(
                image_path,
                json_path,
                raw_path,
                processor,
                model,
                device,
            )

            success += 1
            valid_node_counts.append(
                int(stats["valid_nodes"])
            )

            append_log({
                "filename": fname,
                "status": "ok",
                "json_path": json_path,
                "raw_path": raw_path,
                "error": "",
                **stats,
            })

        except Exception as e:
            failed += 1

            append_log({
                "filename": fname,
                "status": "failed",
                "json_path": json_path,
                "raw_path": raw_path,
                "error": repr(e),
            })

            print(
                f"\n[ERROR] {fname}: {e}\n"
                f"{traceback.format_exc()}"
            )

        if torch.cuda.is_available():
            # 不每张图 empty_cache，避免显著拖慢；
            # 此处仅让 Python 有机会回收临时对象
            pass

    print("\n" + "=" * 70)
    print("Finished")
    print("=" * 70)
    print(f"Success: {success}")
    print(f"Failed : {failed}")

    if valid_node_counts:
        mean_nodes = sum(valid_node_counts) / len(valid_node_counts)
        print(f"Mean ScreenVLM valid nodes/image: {mean_nodes:.3f}")

    print(f"JSON : {OUTPUT_JSON_DIR}")
    print(f"Raw  : {OUTPUT_RAW_DIR}")
    print(f"Log  : {LOG_CSV}")

    print("\n[Sanity check]")
    print("先用这批结果跑你的 structural_evaluator，重点看：")
    print("  common.num_gt_nodes")
    print("  common.num_pred_nodes")
    print("  common.num_matched_nodes")
    print("  common.node_count_recall")
    print("  common.bounding_box_iou")
    print("  PCP")
    print("\n若 50 张结果合理，再把 MAX_IMAGES = None 跑完整测试集。")


if __name__ == "__main__":
    main()