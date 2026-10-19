# HCA-UI: Hierarchy-Constrained Attention for Diffusion-Based Mobile UI Prototyping

Official implementation accompanying **“Hierarchy-Constrained Attention for Diffusion-Based Mobile UI Prototyping”** by Kun Xu, Shang Zheng, Hualong Yu, and Boyin Jin.

## Overview

**HCA-UI** is a hierarchy-aware diffusion framework for mobile UI prototyping from natural-language descriptions and component hierarchies. Built on **Stable Diffusion v1.5**, it introduces **Hierarchy-Constrained Attention (HCA)** to guide visual feature sharing according to component containment and spatial occupancy. The pretrained backbone is frozen while the hierarchy-related modules are trained.

The component hierarchy is also retained as an editable JSON representation for layout correction and localized prototype editing. The diffusion model generates UI images; it does **not** predict the JSON hierarchy.

## Method

- **HierAdapter:** Encodes component geometry, depth, and identity into multiscale spatial features.
- **Parent-exclusive context:** Extracts shared parent-container information from regions not occupied by direct children.
- **Hierarchy-constrained attention:** Routes parent-specific context to direct children while retaining global attention.
- **Adaptive feature fusion:** Balances parent-specific and global information using component depth and area.
- **Structured revision:** Supports constraint-aware layout correction and localized edits using the retained component hierarchy.

## Installation

```bash
git clone https://github.com/ffdsj/HCA-UI.git
cd HCA-UI/sd-hca
conda env create -f environment.yml
conda activate sca-dm
```

The provided `environment.yml` may require platform-specific adjustments.

### Pretrained checkpoint

Obtain the **Stable Diffusion v1.5** checkpoint `v1-5-pruned.ckpt` from an authorized source and place it at:

```text
sd-hca/models/stable-diffusion-v1-1/v1-5-pruned.ckpt
```

The checkpoint is not included in this repository. Configure the checkpoint path in `configs/latent-diffusion/txt2img-1p4B-eval.yaml` if necessary.

## Dataset

This project uses a processed version of the [Rico dataset](https://www.interactionmining.org/archive/rico), with selected mobile UI screenshots, component hierarchies, additional natural-language descriptions, and interface categories.

**Dataset on Hugging Face:** [ff856/HCA-UI-Dataset](https://huggingface.co/datasets/ff856/HCA-UI-Dataset)

The dataset derives in part from Rico. Please consult the original dataset's terms and attribution requirements when reusing or redistributing its screenshots and annotations. The research also evaluates cross-domain transfer on WebUI-7k.

## Training

Run from the `sd-hca/` directory:

```bash
python main.py --train --base configs/latent-diffusion/txt2img-1p4B-eval.yaml
```

Before training, configure the image, JSON, and text dataset paths in the YAML file. For manuscript-specific hyperparameters and experimental details, refer to the paper.

## Inference

Run from the `sd-hca/` directory:

```bash
python inference/inference.py
```

Set the trained checkpoint path and the input/output directories in `inference/inference.py` before running. The inference script is stored under `inference/`, rather than at the repository root.

## Citation

If you use this project in your research, please cite:

```bibtex
@misc{xu_hca_ui,
  title  = {Hierarchy-Constrained Attention for Diffusion-Based Mobile UI Prototyping},
  author = {Xu, Kun and Zheng, Shang and Yu, Hualong and Jin, Boyin},
  note   = {Manuscript; publication details to be added}
}
```

## Acknowledgments

This project builds on [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion) and [Stable Diffusion](https://github.com/CompVis/stable-diffusion), and uses data derived from [Rico](https://www.interactionmining.org/archive/rico). Please respect the applicable upstream licenses and usage terms.
