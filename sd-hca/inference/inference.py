# -*- coding: utf-8 -*-

import json
import math
import random
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchvision.utils import save_image
from tqdm import tqdm

# ============================================================
# 你的项目模块
# ============================================================

from ldm.models.diffusion.ddimhier import DDIMSamplerHier

from ldm.modules.attention import (
    replace_blocks_with_hierarchical,
    inject_depth_map,
)

from dataset import create_maps_from_ui_tree
from ldm.util import instantiate_from_config
from json_parse import parse_json_tree


# ============================================================
# 1. 全局配置
# ============================================================

CONFIG_PATH = (
    "configs/latent-diffusion/"
    "txt2img-1p4B-eval.yaml"
)

CKPT_PATH = (
    r""
    r""
    r""
)


JSON_DIR = Path(
    r""
)

IMAGE_DIR = Path(
    r""
)

CAPTION_DIR = Path(
    r""
)


CAPTION_INDEX_PATH = None


OUTPUT_ROOT = Path(
    r""
    r""
)


# ------------------------------------------------------------
# 推理参数
# ------------------------------------------------------------

DEVICE = "cuda"

DDIM_STEPS = 50

CFG_SCALE = 7.5

BASE_SEED = 20260916


STRICT_PAIRING = True


# ------------------------------------------------------------
# 批量实验建议 False。
#
# Attention 图非常占空间。
# ------------------------------------------------------------

VISUALIZE_ATTENTION = False


# ------------------------------------------------------------
# None = 全部
#
# 测试程序时可设置：
#
# MAX_SAMPLES = 10
# ------------------------------------------------------------

MAX_SAMPLES = 30


# ------------------------------------------------------------
# 已经生成过的样本是否跳过
# ------------------------------------------------------------

RESUME = True


SUPPORTED_IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
]


# ============================================================
# 2. 工具函数
# ============================================================

def set_seed(seed: int):
    """
    设置随机种子。
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def safe_int(value, default=None):
    """
    尽可能将 value 转成 int。
    """

    if value is None:
        return default

    try:
        return int(value)
    except Exception:
        return default


def get_record_id(record):
    """
    从 caption json/jsonl 中寻找样本 ID。
    """

    possible_keys = [
        "sample_id",
        "screen_id",
        "id",
        "image_id",
        "file_id",
        "filename",
        "file_name",
    ]

    for key in possible_keys:

        value = record.get(key)

        if value is None:
            continue

        value = str(value)

        # 如果保存的是 28.png
        value = Path(value).stem

        return value

    return None


def get_record_caption(record):
    """
    从 caption record 中寻找自然语言。
    """

    possible_keys = [
        "caption",
        "prompt",
        "text_prompt",
        "description",
        "text",
    ]

    for key in possible_keys:

        value = record.get(key)

        if isinstance(value, str):

            value = value.strip()

            if value:
                return value

    return None


# ============================================================
# 3. Caption 索引读取
# ============================================================

def load_caption_index(path):
    """
    支持：
        captions.json
        captions.jsonl

    JSON 可以是：

    {
        "28": "caption...",
        "29": "caption..."
    }

    也可以是：

    [
        {
            "sample_id": "28",
            "caption": "..."
        }
    ]
    """

    if path is None:
        return {}

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Caption index 不存在: {path}"
        )

    caption_map = {}

    # ========================================================
    # JSONL
    # ========================================================

    if path.suffix.lower() == ".jsonl":

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            for line_no, line in enumerate(
                f,
                start=1
            ):

                line = line.strip()

                if not line:
                    continue

                record = json.loads(line)

                sample_id = get_record_id(
                    record
                )

                caption = get_record_caption(
                    record
                )

                if sample_id is None:
                    continue

                if caption is None:
                    continue

                if sample_id in caption_map:

                    raise ValueError(
                        f"Caption ID 重复: "
                        f"{sample_id}, "
                        f"line={line_no}"
                    )

                caption_map[sample_id] = caption

    # ========================================================
    # JSON
    # ========================================================

    elif path.suffix.lower() == ".json":

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        # ----------------------------------------------------
        # 格式1
        #
        # {
        #   "28": "caption"
        # }
        # ----------------------------------------------------

        if isinstance(data, dict):

            for key, value in data.items():

                if isinstance(value, str):

                    caption_map[
                        str(key)
                    ] = value.strip()

                elif isinstance(value, dict):

                    sample_id = (
                        get_record_id(value)
                        or str(key)
                    )

                    caption = (
                        get_record_caption(value)
                    )

                    if caption:
                        caption_map[
                            sample_id
                        ] = caption


        elif isinstance(data, list):

            for record in data:

                if not isinstance(
                    record,
                    dict
                ):
                    continue

                sample_id = get_record_id(
                    record
                )

                caption = get_record_caption(
                    record
                )

                if sample_id and caption:

                    if (
                        sample_id
                        in caption_map
                    ):

                        raise ValueError(
                            "Caption ID 重复: "
                            f"{sample_id}"
                        )

                    caption_map[
                        sample_id
                    ] = caption

    else:

        raise ValueError(
            "Caption index 仅支持 "
            ".json 或 .jsonl"
        )

    return caption_map


# ============================================================
# 4. 查找对应图片
# ============================================================

def find_reference_image(
    sample_id,
    image_dir
):
    """
    严格按文件名匹配：

        28.json

    对应：

        28.png

    或：

        28.jpg
        28.jpeg
        28.webp
    """

    matches = []

    for ext in SUPPORTED_IMAGE_EXTENSIONS:

        path = (
            image_dir /
            f"{sample_id}{ext}"
        )

        if path.exists():
            matches.append(path)

    if len(matches) == 0:
        return None

    if len(matches) > 1:

        raise RuntimeError(
            f"{sample_id} 找到多个对应图片："
            f"{matches}"
        )

    return matches[0]


# ============================================================
# 5. 获取自然语言
# ============================================================

def find_caption(
    sample_id,
    json_data,
    caption_dir,
    caption_index
):
    """
    优先级：

    1. sample_id.txt
    2. caption index
    3. JSON 根节点内部 caption/prompt/description
    4. activity 内部字段
    """

    # ========================================================
    # 方式1：同名 txt
    # ========================================================

    txt_path = (
        caption_dir /
        f"{sample_id}.txt"
    )

    if txt_path.exists():

        text = txt_path.read_text(
            encoding="utf-8"
        ).strip()

        if text:

            return (
                text,
                str(txt_path)
            )

    # ========================================================
    # 方式2：统一 caption 文件
    # ========================================================

    if sample_id in caption_index:

        return (
            caption_index[sample_id],
            (
                "caption_index:"
                f"{CAPTION_INDEX_PATH}"
            )
        )

    # ========================================================
    # 方式3：JSON 自带
    # ========================================================

    possible_keys = [
        "caption",
        "prompt",
        "text_prompt",
        "description",
    ]

    for key in possible_keys:

        value = json_data.get(key)

        if (
            isinstance(value, str)
            and value.strip()
        ):

            return (
                value.strip(),
                f"json:{key}"
            )

    # ========================================================
    # activity 中查找
    # ========================================================

    activity = json_data.get(
        "activity"
    )

    if isinstance(activity, dict):

        for key in possible_keys:

            value = activity.get(key)

            if (
                isinstance(value, str)
                and value.strip()
            ):

                return (
                    value.strip(),
                    f"json:activity.{key}"
                )

    return None, None


# ============================================================
# 6. 构建严格的一一对应样本
# ============================================================

def collect_samples():

    caption_index = load_caption_index(
        CAPTION_INDEX_PATH
    )

    json_files = []

    for path in JSON_DIR.glob("*.json"):

        stem = path.stem

        # 排除辅助文件和以前生成的结果
        if stem.endswith("_bb"):
            continue

        if stem.endswith("_generated"):
            continue

        json_files.append(path)

    # ========================================================
    # 尽可能按数字 ID 排序
    # ========================================================

    def sort_key(path):

        stem = path.stem

        if stem.isdigit():
            return (0, int(stem))

        return (1, stem)

    json_files = sorted(
        json_files,
        key=sort_key
    )

    samples = []

    errors = []

    for json_path in json_files:

        sample_id = json_path.stem

        # ----------------------------------------------------
        # JSON
        # ----------------------------------------------------

        try:

            with open(
                json_path,
                "r",
                encoding="utf-8"
            ) as f:

                json_data = json.load(f)

        except Exception as e:

            errors.append(
                {
                    "sample_id":
                        sample_id,

                    "reason":
                        "invalid_json",

                    "detail":
                        str(e),
                }
            )

            continue

        # ----------------------------------------------------
        # 图片
        # ----------------------------------------------------

        try:

            image_path = (
                find_reference_image(
                    sample_id,
                    IMAGE_DIR
                )
            )

        except Exception as e:

            errors.append(
                {
                    "sample_id":
                        sample_id,

                    "reason":
                        "multiple_images",

                    "detail":
                        str(e),
                }
            )

            continue

        if image_path is None:

            errors.append(
                {
                    "sample_id":
                        sample_id,

                    "reason":
                        "missing_image",
                }
            )

            continue

        # ----------------------------------------------------
        # Caption
        # ----------------------------------------------------

        caption, caption_source = (
            find_caption(
                sample_id=sample_id,
                json_data=json_data,
                caption_dir=CAPTION_DIR,
                caption_index=caption_index,
            )
        )

        if not caption:

            errors.append(
                {
                    "sample_id":
                        sample_id,

                    "reason":
                        "missing_caption",
                }
            )

            continue

        # ----------------------------------------------------
        # 一一对应记录
        # ----------------------------------------------------

        samples.append(
            {
                "sample_id":
                    sample_id,

                "caption":
                    caption,

                "caption_source":
                    caption_source,

                "json_path":
                    json_path.resolve(),

                "image_path":
                    image_path.resolve(),
            }
        )

    # ========================================================
    # 保存匹配检查结果
    # ========================================================

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    report = {
        "json_total":
            len(json_files),

        "valid_pairs":
            len(samples),

        "invalid_pairs":
            len(errors),

        "errors":
            errors,
    }

    report_path = (
        OUTPUT_ROOT /
        "pairing_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # 严格模式
    # ========================================================

    if (
        STRICT_PAIRING
        and errors
    ):

        print(
            "\n发现无法一一对应的样本。"
        )

        print(
            f"有效：{len(samples)}"
        )

        print(
            f"异常：{len(errors)}"
        )

        print(
            f"检查文件：{report_path}"
        )

        raise RuntimeError(
            "STRICT_PAIRING=True，"
            "请先解决数据对应关系问题。"
        )

    return samples


# ============================================================
# 7. 推理模型
# ============================================================

class UIHierarchyInference:

    def __init__(
        self,
        config_path,
        ckpt_path,
        device="cuda"
    ):

        self.device = device

        # ====================================================
        # 配置
        # ====================================================

        self.config = OmegaConf.load(
            config_path
        )

        print(
            "Initializing model..."
        )

        self.model = (
            instantiate_from_config(
                self.config.model
            )
        )

        # ====================================================
        # 替换层级 Transformer block
        # ====================================================

        replace_blocks_with_hierarchical(
            self.model.model.diffusion_model
        )

        # ====================================================
        # 加载 checkpoint
        # ====================================================

        print(
            f"Loading checkpoint: "
            f"{ckpt_path}"
        )

        checkpoint = torch.load(
            ckpt_path,
            map_location="cpu"
        )

        if (
            isinstance(checkpoint, dict)
            and "state_dict" in checkpoint
        ):

            state_dict = (
                checkpoint["state_dict"]
            )

        else:

            state_dict = checkpoint

        result = (
            self.model.load_state_dict(
                state_dict,
                strict=False
            )
        )

        print(
            f"Missing keys: "
            f"{len(result.missing_keys)}"
        )

        print(
            f"Unexpected keys: "
            f"{len(result.unexpected_keys)}"
        )

        self.model = (
            self.model
            .to(device)
            .eval()
        )

        self.sampler = (
            DDIMSamplerHier(
                self.model
            )
        )

        print(
            "Model ready!"
        )

    # ========================================================
    # JSON 分辨率
    # ========================================================

    def _get_json_resolution(
        self,
        json_data
    ):

        def get_bounds_from_node(
            node
        ):

            if not isinstance(
                node,
                dict
            ):
                return None

            bounds = node.get(
                "bounds"
            )

            if (
                bounds
                and len(bounds) == 4
            ):

                w = bounds[2]
                h = bounds[3]

                if (
                    w > 0
                    and h > 0
                ):

                    return (
                        w,
                        h
                    )

            return None

        # ----------------------------------------------------
        # Rico activity.root
        # ----------------------------------------------------

        if (
            "activity" in json_data
            and isinstance(
                json_data["activity"],
                dict
            )
        ):

            root = (
                json_data[
                    "activity"
                ].get(
                    "root",
                    {}
                )
            )

        else:

            root = json_data

        # ----------------------------------------------------
        # 根节点
        # ----------------------------------------------------

        result = (
            get_bounds_from_node(
                root
            )
        )

        if result:
            return result

        # ----------------------------------------------------
        # 子节点回退
        # ----------------------------------------------------

        children = root.get(
            "children",
            []
        )

        for child in children:

            result = (
                get_bounds_from_node(
                    child
                )
            )

            if result:

                print(
                    "[WARN] 使用子节点"
                    f"分辨率: {result}"
                )

                return result

        print(
            "[WARN] JSON 无有效 "
            "bounds，使用 "
            "1440x2560"
        )

        return (
            1440,
            2560
        )

    # ========================================================
    # 获取节点数字 ID
    # ========================================================

    def _build_node_lookup(
        self,
        nodes
    ):

        result = {}

        if not nodes:
            return result

        possible_id_keys = [
            "ancestry_id",
            "node_id",
            "map_id",
            "id",
        ]

        for node in nodes:

            if not isinstance(
                node,
                dict
            ):
                continue

            for key in possible_id_keys:

                if key not in node:
                    continue

                value = safe_int(
                    node.get(key)
                )

                if value is not None:

                    result[value] = node

                    break

        return result

    # ========================================================
    # 获取文字
    # ========================================================

    def _extract_text(
        self,
        node
    ):

        if not isinstance(
            node,
            dict
        ):
            return ""

        possible_keys = [
            "text",
            "label",
            "content",
            "contentDescription",
            "content-desc",
            "description",
            "hint",
        ]

        for key in possible_keys:

            value = node.get(key)

            if (
                isinstance(value, str)
                and value.strip()
            ):

                return value.strip()

        return ""

    # ========================================================
    # 类型映射
    # ========================================================

    def _map_component_type(
        self,
        raw_type
    ):

        if raw_type is None:
            return "View"

        raw_type = str(
            raw_type
        ).lower()

        mapping = {
            "button": "Button",
            "btn": "Button",

            "text": "Text",
            "label": "Text",
            "textview": "Text",

            "input": "Input",
            "edittext": "Input",

            "image": "Image",
            "img": "Image",
            "imageview": "Image",

            "card": "Card",

            "container": "Container",
            "viewgroup": "Container",

            "root": "Root",

            "toolbar": "Toolbar",
            "header": "Header",

            "list": "List",
            "listview": "List",

            "item": "ListItem",
        }

        return mapping.get(
            raw_type,
            "View"
        )

    # ========================================================
    # 从生成图像提取组件颜色
    # ========================================================

    def _extract_region_color(
        self,
        image_np,
        latent_mask
    ):
        """
        image_np:
            [3, H, W], range 0~1

        latent_mask:
            [64,64] bool

        使用 nearest 上采样 mask，
        只统计该组件真正对应区域。
        """

        img_h = image_np.shape[1]
        img_w = image_np.shape[2]

        mask_tensor = torch.from_numpy(
            latent_mask.astype(
                np.float32
            )
        )

        mask_tensor = (
            mask_tensor[
                None,
                None,
                ...
            ]
        )

        mask_high = F.interpolate(
            mask_tensor,
            size=(
                img_h,
                img_w
            ),
            mode="nearest"
        )

        mask_high = (
            mask_high[
                0,
                0
            ]
            .numpy()
            > 0.5
        )

        if not mask_high.any():

            return (
                "#ffffff",
                "#ffffff"
            )

        # [3, N]
        pixels = (
            image_np[
                :,
                mask_high
            ]
        )

        mean_rgb = np.mean(
            pixels,
            axis=1
        )

        median_rgb = np.median(
            pixels,
            axis=1
        )

        mean_values = [
            int(
                np.clip(
                    x * 255,
                    0,
                    255
                )
            )
            for x in mean_rgb
        ]

        median_values = [
            int(
                np.clip(
                    x * 255,
                    0,
                    255
                )
            )
            for x in median_rgb
        ]

        mean_hex = (
            "#{:02x}{:02x}{:02x}"
        ).format(
            *mean_values
        )

        median_hex = (
            "#{:02x}{:02x}{:02x}"
        ).format(
            *median_values
        )

        return (
            mean_hex,
            median_hex
        )

    # ========================================================
    # 保存结构化 JSON
    # ========================================================

    def _save_structured_output(
        self,
        x_samples,
        ancestry_map,
        parent_map,
        nodes,
        caption,
        output_path,
        sample_id=None,
        input_json_path=None,
        reference_image_path=None,
        seed=None,
    ):

        output_path = Path(
            output_path
        )

        json_path = (
            output_path
            .with_suffix(".json")
        )

        # ----------------------------------------------------
        # CFG 时 parent_map 有 2 份
        # ----------------------------------------------------

        if isinstance(
            parent_map,
            list
        ):

            parent_dict = (
                parent_map[0]
            )

        else:

            parent_dict = (
                parent_map
            )

        if parent_dict is None:
            parent_dict = {}

        # ----------------------------------------------------
        # ancestry map
        # ----------------------------------------------------

        if ancestry_map.dim() == 4:

            ancestry_np = (
                ancestry_map[
                    0,
                    0
                ]
                .detach()
                .cpu()
                .numpy()
            )

        elif ancestry_map.dim() == 3:

            ancestry_np = (
                ancestry_map[
                    0
                ]
                .detach()
                .cpu()
                .numpy()
            )

        else:

            ancestry_np = (
                ancestry_map
                .detach()
                .cpu()
                .numpy()
            )

        latent_h, latent_w = (
            ancestry_np.shape
        )

        # ----------------------------------------------------
        # 最终生成图片
        # ----------------------------------------------------

        image_np = (
            x_samples[
                0
            ]
            .detach()
            .cpu()
            .numpy()
        )

        img_h = (
            x_samples.shape[2]
        )

        img_w = (
            x_samples.shape[3]
        )

        # ----------------------------------------------------
        # node lookup
        # ----------------------------------------------------

        node_lookup = (
            self._build_node_lookup(
                nodes
            )
        )

        # ----------------------------------------------------
        # JSON 主结构
        # ----------------------------------------------------

        data = {

            "metadata": {

                "sample_id":
                    sample_id,

                "model":
                    "Hierarchical-SD-v1",

                "timestamp":
                    datetime.now()
                    .isoformat(),

                "prompt":
                    caption,

                "seed":
                    seed,

                "hierarchy_constraint":
                    True,

                "gate_mechanism":
                    "sigmoid_gated_fusion",

                "input_json":
                    (
                        str(
                            input_json_path
                        )
                        if input_json_path
                        else None
                    ),

                "reference_image":
                    (
                        str(
                            reference_image_path
                        )
                        if reference_image_path
                        else None
                    ),

                "generated_image":
                    str(
                        output_path
                    ),

                "color_source":
                    (
                        "decoded_diffusion_image"
                    ),

                "color_extraction":
                    (
                        "component_mask_mean_"
                        "and_median_rgb"
                    ),
            },

            "canvas": {

                "width":
                    int(img_w),

                "height":
                    int(img_h),

                "coordinate_system":
                    "ltwh",
            },

            "components": [],

            "relationships": {
                "hierarchy_edges": []
            },
        }

        # ----------------------------------------------------
        # 所有组件 ID
        # ----------------------------------------------------

        unique_ids = np.unique(
            ancestry_np
        )

        unique_ids = [
            int(x)
            for x in unique_ids
            if x > 0
        ]

        # ====================================================
        # 遍历所有组件
        # ====================================================

        for cid in unique_ids:

            mask = (
                ancestry_np
                == cid
            )

            y_coords, x_coords = (
                np.where(mask)
            )

            if len(
                y_coords
            ) == 0:

                continue

            # ------------------------------------------------
            # latent bbox
            # ------------------------------------------------

            lx1 = int(
                x_coords.min()
            )

            ly1 = int(
                y_coords.min()
            )

            lx2 = int(
                x_coords.max()
                + 1
            )

            ly2 = int(
                y_coords.max()
                + 1
            )

            # ------------------------------------------------
            # 映射到最终图片
            #
            # 不要求一定是 8 倍，
            # 因此避免 img_h // 64。
            # ------------------------------------------------

            x1 = int(
                round(
                    lx1
                    / latent_w
                    * img_w
                )
            )

            y1 = int(
                round(
                    ly1
                    / latent_h
                    * img_h
                )
            )

            x2 = int(
                round(
                    lx2
                    / latent_w
                    * img_w
                )
            )

            y2 = int(
                round(
                    ly2
                    / latent_h
                    * img_h
                )
            )

            x1 = max(
                0,
                min(
                    img_w,
                    x1
                )
            )

            x2 = max(
                0,
                min(
                    img_w,
                    x2
                )
            )

            y1 = max(
                0,
                min(
                    img_h,
                    y1
                )
            )

            y2 = max(
                0,
                min(
                    img_h,
                    y2
                )
            )

            width = max(
                1,
                x2 - x1
            )

            height = max(
                1,
                y2 - y1
            )

            # ------------------------------------------------
            # 从模型解码图像中提取颜色
            # ------------------------------------------------

            (
                mean_color,
                median_color
            ) = (
                self._extract_region_color(
                    image_np,
                    mask
                )
            )

            # ------------------------------------------------
            # 原节点信息
            # ------------------------------------------------

            node_info = (
                node_lookup.get(
                    cid,
                    {}
                )
            )

            raw_type = (
                node_info.get(
                    "componentLabel"
                )
                or
                node_info.get(
                    "type"
                )
                or
                node_info.get(
                    "class"
                )
                or
                f"Component_{cid}"
            )

            component_type = (
                self._map_component_type(
                    raw_type
                )
            )

            text_value = (
                self._extract_text(
                    node_info
                )
            )

            ancestry_id = (
                node_info.get(
                    "ancestry_id",
                    cid
                )
            )

            # ------------------------------------------------
            # 父节点
            # ------------------------------------------------

            parent_value = (
                parent_dict.get(
                    cid,
                    None
                )
            )

            parent_id = safe_int(
                parent_value,
                default=None
            )

            if (
                parent_id is not None
                and parent_id <= 0
            ):

                parent_id = None

            # ------------------------------------------------
            # Component
            # ------------------------------------------------

            component = {

                "id":
                    cid,

                "ancestry_id":
                    ancestry_id,

                "type":
                    component_type,

                "raw_type":
                    str(raw_type),

                # ============================================
                # 后面 E4 修改文字时直接修改这个字段
                # ============================================

                "text":
                    text_value,

                # ============================================
                # 后面 E4 修改位置、尺寸用这个字段
                # ============================================

                "bounds": {

                    "x":
                        int(x1),

                    "y":
                        int(y1),

                    "width":
                        int(width),

                    "height":
                        int(height),
                },

                # 保留 latent 坐标，方便追踪
                "latent_bounds": {

                    "x1":
                        lx1,

                    "y1":
                        ly1,

                    "x2":
                        lx2,

                    "y2":
                        ly2,
                },

                "parent_id":
                    parent_id,

                # ============================================
                # 颜色来自生成图像
                # ============================================

                "style": {

                    "backgroundColor":
                        mean_color,

                    "backgroundColorRobust":
                        median_color,

                    "color_source":
                        (
                            "decoded_"
                            "diffusion_image"
                        ),
                },

                "constraints": {

                    "hierarchy_attention":
                        True,

                    "mask_strict":
                        True,
                },
            }

            data[
                "components"
            ].append(
                component
            )

            # ------------------------------------------------
            # 父子边
            # ------------------------------------------------

            if parent_id is not None:

                data[
                    "relationships"
                ][
                    "hierarchy_edges"
                ].append(
                    {
                        "from":
                            parent_id,

                        "to":
                            cid,

                        "type":
                            "parent-child",
                    }
                )

        # ====================================================
        # 计算几何包含关系
        # ====================================================

        component_map = {
            c["id"]: c
            for c in data[
                "components"
            ]
        }

        for component in data[
            "components"
        ]:

            parent_id = (
                component.get(
                    "parent_id"
                )
            )

            if (
                parent_id is None
                or parent_id
                not in component_map
            ):
                continue

            child_bounds = (
                component[
                    "bounds"
                ]
            )

            parent_bounds = (
                component_map[
                    parent_id
                ][
                    "bounds"
                ]
            )

            valid = (

                child_bounds["x"]
                >=
                parent_bounds["x"]

                and

                child_bounds["y"]
                >=
                parent_bounds["y"]

                and

                (
                    child_bounds["x"]
                    +
                    child_bounds["width"]
                )
                <=
                (
                    parent_bounds["x"]
                    +
                    parent_bounds["width"]
                )

                and

                (
                    child_bounds["y"]
                    +
                    child_bounds["height"]
                )
                <=
                (
                    parent_bounds["y"]
                    +
                    parent_bounds["height"]
                )
            )

            component[
                "constraints"
            ][
                "valid_containment"
            ] = bool(valid)

        # ====================================================
        # 保存
        # ====================================================

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "Structured JSON saved: "
            f"{json_path}"
        )

        return json_path

    # ========================================================
    # Attention 可视化
    # ========================================================

    def _visualize_attention(
        self,
        output_path
    ):

        import matplotlib.pyplot as plt

        global_attns = []
        parent_attns = []

        for (
            name,
            module
        ) in (
            self.model
            .model
            .diffusion_model
            .named_modules()
        ):

            if (
                hasattr(
                    module,
                    "global_attn"
                )
                and
                module.global_attn
                is not None
            ):

                global_attns.append(
                    module
                    .global_attn
                    .detach()
                    .cpu()
                    .numpy()
                )

            if (
                hasattr(
                    module,
                    "parent_attn"
                )
                and
                module.parent_attn
                is not None
            ):

                parent_attns.append(
                    module
                    .parent_attn
                    .detach()
                    .cpu()
                    .numpy()
                )

        if not parent_attns:
            return

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 6)
        )

        def prepare_map(arr):

            arr = np.asarray(arr)

            while (
                arr.ndim > 1
                and arr.shape[0] > 1
            ):
                arr = arr.mean(
                    axis=0
                )

            arr = np.squeeze(arr)

            if arr.ndim == 1:

                side = int(
                    math.sqrt(
                        arr.size
                    )
                )

                if (
                    side * side
                    == arr.size
                ):

                    arr = arr.reshape(
                        side,
                        side
                    )

            return arr

        if global_attns:

            g = prepare_map(
                global_attns[-1]
            )

            axes[0].imshow(
                g,
                cmap="hot"
            )

            axes[0].set_title(
                "Global Self-Attention"
            )

            axes[0].axis(
                "off"
            )

        p = prepare_map(
            parent_attns[-1]
        )

        axes[1].imshow(
            p,
            cmap="hot"
        )

        axes[1].set_title(
            "Parent Hierarchy Attention"
        )

        axes[1].axis(
            "off"
        )

        plt.tight_layout()

        output_path = Path(
            output_path
        )

        vis_path = (
            output_path.parent /
            (
                output_path.stem
                +
                "_attention.png"
            )
        )

        plt.savefig(
            vis_path,
            dpi=150,
            bbox_inches="tight"
        )

        plt.close()

        print(
            "Attention map saved: "
            f"{vis_path}"
        )

    # ========================================================
    # Gate log
    # ========================================================

    def _save_gate_log(
        self,
        output_path
    ):

        from ldm.modules.attention import (
            HierarchicalBasicTransformerBlock
        )

        records = (
            HierarchicalBasicTransformerBlock
            .gate_records
        )

        if not records:
            return None

        output_path = Path(
            output_path
        )

        log_path = (
            output_path.parent /
            (
                output_path.stem
                +
                "_gate_log.txt"
            )
        )

        with open(
            log_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                f"{'t':>4} | "
                f"{'phase':>9} | "
                f"{'gate':>7} | "
                f"block\n"
            )

            f.write(
                "-" * 45
                + "\n"
            )

            for record in records:

                f.write(
                    f"{record['t']:4d} | "
                    f"{record['phase']:9s} | "
                    f"{record['gate']:7.4f} | "
                    f"{record['block_id']:04x}"
                    f"\n"
                )

            structure_gates = [
                r["gate"]
                for r in records
                if (
                    r["phase"]
                    == "structure"
                )
            ]

            style_gates = [
                r["gate"]
                for r in records
                if (
                    r["phase"]
                    == "style"
                )
            ]

            f.write(
                "-" * 45
                + "\n"
            )

            if structure_gates:

                f.write(
                    "structure phase "
                    "mean gate = "
                    f"{np.mean(structure_gates):.4f}"
                    "\n"
                )

            if style_gates:

                f.write(
                    "style phase "
                    "mean gate = "
                    f"{np.mean(style_gates):.4f}"
                    "\n"
                )

        records.clear()

        return log_path

    # ========================================================
    # 单样本生成
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        caption,
        json_path,
        output_path,
        sample_id=None,
        reference_image_path=None,
        ddim_steps=50,
        scale=7.5,
        seed=0,
        visualize_attention=False,
    ):

        set_seed(
            seed
        )

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ====================================================
        # 每次生成前清空 gate records
        # ====================================================

        try:

            from ldm.modules.attention import (
                HierarchicalBasicTransformerBlock
            )

            HierarchicalBasicTransformerBlock\
                .gate_records\
                .clear()

        except Exception:
            pass

        # ====================================================
        # Text conditioning
        # ====================================================

        c = (
            self.model
            .get_learned_conditioning(
                [caption]
            )
        )

        uc = (
            self.model
            .get_learned_conditioning(
                [""]
            )
        )

        # ====================================================
        # JSON
        # ====================================================

        json_path = Path(
            json_path
        )

        if not json_path.exists():

            raise FileNotFoundError(
                f"JSON 不存在: "
                f"{json_path}"
            )

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as f:

            json_data = json.load(f)

        nodes, _ = (
            parse_json_tree(
                json_data
            )
        )

        json_w, json_h = (
            self._get_json_resolution(
                json_data
            )
        )

        # ====================================================
        # 生成 hierarchy maps
        # ====================================================

        (
            depth_map,
            ancestry_map,
            parent_dict
        ) = (
            create_maps_from_ui_tree(
                nodes,
                img_wh=(
                    json_w,
                    json_h
                ),
                latent_size=64,
            )
        )

        depth_map = (
            depth_map
            .unsqueeze(0)
            .to(self.device)
        )

        ancestry_map = (
            ancestry_map
            .unsqueeze(0)
            .to(self.device)
        )

        parent_map = [
            parent_dict
        ]

        # ====================================================
        # CFG batch 扩展
        # ====================================================

        if scale != 1.0:

            depth_map = torch.cat(
                [
                    depth_map,
                    depth_map,
                ],
                dim=0
            )

            ancestry_map = torch.cat(
                [
                    ancestry_map,
                    ancestry_map,
                ],
                dim=0
            )

            parent_map = (
                parent_map
                +
                parent_map
            )

        # ====================================================
        # 注入结构条件
        # ====================================================

        inject_depth_map(
            self.model
            .model
            .diffusion_model,

            depth_map,

            ancestry_map,

            parent_map,
        )

        # ====================================================
        # Gate 检查
        # ====================================================

        gates = []

        for (
            name,
            module
        ) in (
            self.model
            .model
            .diffusion_model
            .named_modules()
        ):

            if hasattr(
                module,
                "gate_weight"
            ):

                gate = (
                    torch.sigmoid(
                        module.gate_weight
                    )
                    .detach()
                    .item()
                )

                gates.append(
                    gate
                )

        if gates:

            print(
                "Gate mean: "
                f"{np.mean(gates):.4f}"
            )

        # ====================================================
        # latent shape
        # ====================================================

        shape = (
            self.model.channels,
            self.model.image_size,
            self.model.image_size,
        )

        print(
            f"Generating ID="
            f"{sample_id}"
        )

        print(
            f"Seed={seed}, "
            f"steps={ddim_steps}, "
            f"CFG={scale}"
        )

        # ====================================================
        # DDIM
        # ====================================================

        samples, intermediates = (
            self.sampler.sample(

                S=ddim_steps,

                batch_size=1,

                shape=shape,

                conditioning=c,

                unconditional_conditioning=uc,

                unconditional_guidance_scale=scale,

                use_hier=True,

                depth_map=depth_map,

                ancestry_map=ancestry_map,

                parent_map=parent_map,

                verbose=False,
            )
        )

        # ====================================================
        # VAE 解码
        # ====================================================

        x_samples = (
            self.model
            .decode_first_stage(
                samples
            )
        )

        x_samples = torch.clamp(
            (
                x_samples
                + 1.0
            )
            / 2.0,
            min=0.0,
            max=1.0,
        )

        # ====================================================
        # 保存生成图片
        # ====================================================

        save_image(
            x_samples,
            str(output_path)
        )

        print(
            f"Generated image saved: "
            f"{output_path}"
        )

        # ====================================================
        # 保存输出 JSON
        # ====================================================

        json_output_path = (
            self._save_structured_output(

                x_samples=x_samples,

                ancestry_map=ancestry_map,

                parent_map=parent_map,

                nodes=nodes,

                caption=caption,

                output_path=output_path,

                sample_id=sample_id,

                input_json_path=
                    json_path,

                reference_image_path=
                    reference_image_path,

                seed=seed,
            )
        )

        # ====================================================
        # Attention
        # ====================================================

        if visualize_attention:

            try:

                self._visualize_attention(
                    output_path
                )

            except Exception as e:

                print(
                    "[WARN] Attention "
                    "visualization failed: "
                    f"{e}"
                )

        # ====================================================
        # Gate log
        # ====================================================

        gate_log_path = (
            self._save_gate_log(
                output_path
            )
        )

        return {
            "generated_image":
                str(output_path),

            "generated_json":
                str(json_output_path),

            "gate_log":
                (
                    str(gate_log_path)
                    if gate_log_path
                    else None
                ),
        }


# ============================================================
# 8. JSONL 写入
# ============================================================

def append_jsonl(
    path,
    record
):

    with open(
        path,
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            )
            +
            "\n"
        )


# ============================================================
# 9. Batch
# ============================================================

def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # 配对
    # ========================================================

    samples = collect_samples()

    if MAX_SAMPLES is not None:

        samples = samples[
            :MAX_SAMPLES
        ]

    print(
        "=" * 70
    )

    print(
        "One-to-one paired samples: "
        f"{len(samples)}"
    )

    print(
        "=" * 70
    )

    if not samples:

        print(
            "没有可生成样本。"
        )

        return

    # ========================================================
    # 输出记录
    # ========================================================

    manifest_path = (
        OUTPUT_ROOT /
        "generation_manifest.jsonl"
    )

    failed_path = (
        OUTPUT_ROOT /
        "generation_failed.jsonl"
    )

    # 每次运行重新建立
    manifest_path.write_text(
        "",
        encoding="utf-8"
    )

    failed_path.write_text(
        "",
        encoding="utf-8"
    )

    # ========================================================
    # 模型只加载一次
    # ========================================================

    inference = (
        UIHierarchyInference(
            config_path=
                CONFIG_PATH,

            ckpt_path=
                CKPT_PATH,

            device=
                DEVICE,
        )
    )

    success_count = 0
    failed_count = 0
    skipped_count = 0

    # ========================================================
    # 开始批量生成
    # ========================================================

    for index, sample in enumerate(
        tqdm(
            samples,
            desc="Batch generation"
        )
    ):

        sample_id = (
            sample[
                "sample_id"
            ]
        )

        caption = (
            sample[
                "caption"
            ]
        )

        input_json = Path(
            sample[
                "json_path"
            ]
        )

        reference_image = Path(
            sample[
                "image_path"
            ]
        )

        seed = (
            BASE_SEED
            +
            index
        )

        # ----------------------------------------------------
        # 每个 ID 单独目录
        # ----------------------------------------------------

        sample_dir = (
            OUTPUT_ROOT /
            sample_id
        )

        sample_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        generated_image = (
            sample_dir /
            (
                f"{sample_id}"
                "_generated.png"
            )
        )

        generated_json = (
            sample_dir /
            (
                f"{sample_id}"
                "_generated.json"
            )
        )

        # ====================================================
        # Manifest 基础信息
        # ====================================================

        record = {

            "index":
                index,

            "sample_id":
                sample_id,

            "caption":
                caption,

            "caption_source":
                sample[
                    "caption_source"
                ],

            "input_json":
                str(input_json),

            "reference_image":
                str(reference_image),

            "generated_image":
                str(
                    generated_image
                ),

            "generated_json":
                str(
                    generated_json
                ),

            "seed":
                seed,

            "ddim_steps":
                DDIM_STEPS,

            "cfg_scale":
                CFG_SCALE,

            "status":
                None,

            "error":
                None,
        }

        # ====================================================
        # Resume
        # ====================================================

        if (
            RESUME
            and generated_image.exists()
            and generated_json.exists()
        ):

            record[
                "status"
            ] = "skipped_existing"

            skipped_count += 1

            append_jsonl(
                manifest_path,
                record
            )

            continue

        # ====================================================
        # Generate
        # ====================================================

        try:

            result = (
                inference.generate(

                    caption=
                        caption,

                    json_path=
                        input_json,

                    output_path=
                        generated_image,

                    sample_id=
                        sample_id,

                    reference_image_path=
                        reference_image,

                    ddim_steps=
                        DDIM_STEPS,

                    scale=
                        CFG_SCALE,

                    seed=
                        seed,

                    visualize_attention=
                        VISUALIZE_ATTENTION,
                )
            )

            # ------------------------------------------------
            # 强制检查核心文件
            # ------------------------------------------------

            if not generated_image.exists():

                raise RuntimeError(
                    "生成图片不存在"
                )

            if not generated_json.exists():

                raise RuntimeError(
                    "生成 JSON 不存在"
                )

            record[
                "status"
            ] = "success"

            success_count += 1

        except Exception as e:

            failed_count += 1

            record[
                "status"
            ] = "failed"

            record[
                "error"
            ] = str(e)

            print(
                "\n"
                + "=" * 70
            )

            print(
                f"[FAILED] "
                f"{sample_id}"
            )

            traceback.print_exc()

            append_jsonl(
                failed_path,
                record
            )

        # ====================================================
        # 每个样本结束后立即写记录
        # ====================================================

        append_jsonl(
            manifest_path,
            record
        )

    # ========================================================
    # 总结
    # ========================================================

    summary = {

        "total":
            len(samples),

        "success":
            success_count,

        "failed":
            failed_count,

        "skipped":
            skipped_count,

        "output_root":
            str(OUTPUT_ROOT),

        "manifest":
            str(manifest_path),

        "failed_file":
            str(failed_path),

        "timestamp":
            datetime.now()
            .isoformat(),
    }

    summary_path = (
        OUTPUT_ROOT /
        "generation_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "Batch generation finished"
    )

    print(
        f"Total:   "
        f"{len(samples)}"
    )

    print(
        f"Success: "
        f"{success_count}"
    )

    print(
        f"Skipped: "
        f"{skipped_count}"
    )

    print(
        f"Failed:  "
        f"{failed_count}"
    )

    print(
        f"Manifest: "
        f"{manifest_path}"
    )

    print(
        f"Summary: "
        f"{summary_path}"
    )

    print(
        "=" * 70
    )


# ============================================================
# 10. Main
# ============================================================

if __name__ == "__main__":

    main()