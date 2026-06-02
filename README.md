# Reproduction of On-the-fly Repulsion for Diffusion Transformers
## Overview

This repository is my reproduction of the paper:

**On-the-fly Repulsion in the Contextual Space for Rich Diversity in Diffusion Transformers**

The implementation is based on the official FLUX repository from Black Forest Labs.


This repository contains a reproduction of Contextual Repulsion implemented on FLUX.1-dev for text-to-image generation.


### Open-weight models

This reproduction mainly uses the following open-weight model:

| Name                        | Usage                                                      | HuggingFace repo                                               | License                                                               |
| --------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------- | --------------------------------------------------------------------- |                  
| `FLUX.1 [dev]`              | [Text to Image](docs/text-to-image.md)                     | https://huggingface.co/black-forest-labs/FLUX.1-dev            | [FLUX.1-dev Non-Commercial License](model_licenses/LICENSE-FLUX1-dev) |

The weights of the autoencoder are also released under [apache-2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) and can be found in the HuggingFace repos above.


## Reproduction Details

This reproduction mainly modifies `model.py` and `sampling.py` to integrate Contextual Repulsion into the FLUX.1-dev generation process.

In `model.py`, Contextual Repulsion is added inside both the dual-stream blocks and the single-stream blocks. During the forward process, when a repulsion configuration is provided, the intermediate contextual tokens are updated by the repulsion operation.

In `sampling.py`, the `denoise()` function is modified to control when Contextual Repulsion is applied. Specifically, the current denoising step is checked by `step_idx < tau`. Only when this condition is satisfied, the repulsion configuration is passed into `model.py`.

The overall control flow is as follows:

1. `generate.py` reads command-line arguments such as `eta`, `inner_steps`, and `tau`.
2. `sampling.py` / `denoise()` checks whether the current denoising step satisfies `step_idx < tau`.
3. `model.py` / `Flux.forward()` applies Contextual Repulsion inside the transformer blocks.

This design enables Contextual Repulsion to be applied only during the first `tau` denoising steps.


## Generation

The main generation script is `generate.py`.

This script is used to generate images with FLUX.1-dev.  
Contextual Repulsion can be enabled by setting a non-zero `eta` value.

Example command:

```bash
python3 generate.py \
  --name flux-dev \
  --prompt_file captions_val2017.json \
  --output_dir ./output \
  --max_prompts 1000 \
  --images_per_prompt 4 \
  --width 512 \
  --height 512 \
  --num_steps 20 \
  --guidance 3.5 \
  --eta 5e9 \
  --contextual_repulsion_inner_steps 50 \
  --contextual_repulsion_tau 1 \
  --save_format png \
  --offload \
  --resume
```

The parameter `eta` controls the strength of Contextual Repulsion.

- `--eta 0`: baseline generation without Contextual Repulsion.
- `--eta > 0`: generation with Contextual Repulsion.



## Evaluation

This repository provides evaluation scripts for reproducing the quantitative metrics reported in the paper.

The evaluation scripts are:

| Script | Metric |
| --- | --- |
| `evaluate_vqa.py` | VQAScore ↑ |
| `evaluate_vd.py` | Vendi Inception Score ↑ |
| `evaluate_irkid.py` | Kernel Inception Distance (KID) ↓ and ImageReward ↑ |

To evaluate VQAScore, run:

```bash
    python evaluate_vqa.py \
  --exp_dirs \
    XXX \
    XXX \
    XXX \
    XXX \
    XXX \
  --vqa_model clip-flant5-xl \
  --vqa_batch_size 16 \
  --output_txt eval_flux_vqa.txt \
```

To evaluate Vendi Inception Score, run:

```bash
python evaluate_vd.py \
  --exp_dirs \
    XXX \
    XXX \
    XXX \
    XXX \
    XXX \
  --device cuda \
  --batch_size 64 \
  --output_txt eval_flux_vendi.txt \
```

To evaluate Kernel Inception Distance and ImageReward, run:

```bash
python evaluate_irkid.py \
  --exp_dirs \
    XXX \
    XXX \
    XXX \
    XXX \
    XXX \
  --kid_reference_dir XXX \
  --device cuda \
  --kid_batch_size 128 \
  --output_txt eval_flux_kid_ir.txt \
```

These metrics correspond to the evaluation protocol used in the reproduced paper.


## Citation

This repository is an unofficial reproduction of the following paper:

**On-the-fly Repulsion in the Contextual Space for Rich Diversity in Diffusion Transformers**

If you use this reproduction in your research, please cite the original paper. Since this implementation is based on the FLUX codebase, please also cite the upstream FLUX repository:

```bib
@article{dahary2026repulsion,
  title   = {On-the-fly Repulsion in the Contextual Space for Rich Diversity in Diffusion Transformers},
  author  = {Dahary, Omer and Koren, Benaya and Garibi, Daniel and Cohen-Or, Daniel},
  journal = {arXiv preprint arXiv:2603.28762},
  year    = {2026},
  doi     = {10.48550/arXiv.2603.28762}
}

@misc{flux2024,
    author={Black Forest Labs},
    title={FLUX},
    year={2024},
    howpublished={\url{https://github.com/black-forest-labs/flux}},
}
```
