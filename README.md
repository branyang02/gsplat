# Installation

tested on Python 3.9.19, CUDA 11.8

1. Clone the repository

```
git clone --recurse-submodules https://github.com/branyang02/gsplat.git
cd gsplat
```

2. Create a new conda environment

```
conda env create -f environment.yml && conda activaate semantic_gen
```

3. Install `gsplat` and `segment-anything-langsplat`

```
pip install -e .
pip install -e segment-anything-langsplat
```

4. Navigate to `semantic_gen` directory and install dependencies

```
cd 3d_semantic && pip install -r requirements.txt
```

5. Download SAM checkpoint

```
mkdir -p ckpts && wget -P ckpts https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

# Usage

All scripts should be run from the `semantic_gen` directory.

Download [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) dataset:

```
python datasets/download_dataset.py
```

## Preprocess Data

First, we extract CLIP embeddings for the training and validation sets.

```
python preprocess.py --data_dir data/[dataset_dir] --sam_ckpt [sam_ckpt] --num_workers 8
```

Adjust `num_workers` based on your GPU's capacity. Data are processed in parallel to speed up the process.

This saves the precomputed embeddings in `data/[dataset_dir]/trainset` and `data/[dataset_dir]/valset`.

## Train

```
python sam_trainer.py --data_dir data/[dataset_dir] --result_dir results/[dataset_dir]
```

## Evaluate and Visualize

```
python sam_viewer.py --ckpt results/[dataset_dir]/ckpts/[ckpt]
```

# gsplat

[![Core Tests.](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml)
[![Docs](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml)

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), but we’ve made gsplat even faster, more memory efficient, and with a growing list of new features!

<div align="center">
  <video src="https://github.com/nerfstudio-project/gsplat/assets/10151885/64c2e9ca-a9a6-4c7e-8d6f-47eeacd15159" width="100%" />
</div>

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easiest way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).

```bash
pip install gsplat
```

Or install from source. In this way it will build the CUDA code during installation.

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

To install gsplat on Windows, please check [this instruction](docs/INSTALL_WIN.md).

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

```bash
# under examples/
pip install -r requirements.txt
# download mipnerf_360 benchmark data
python datasets/download_dataset.py
# run batch evaluation
bash benchmark.sh
```

## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples (requires to install some exta dependences via `pip install -r examples/requirements.txt`)

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)

## Development and Contribution

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the following wonderful contributors (unordered):

- [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (UC Berkeley): Mentor of the project.
- [Matthew Tancik](https://www.matthewtancik.com/about-me) (Luma AI): Mentor of the project.
- [Vickie Ye](https://people.eecs.berkeley.edu/~vye/) (UC Berkeley): Project lead. v0.1 lead.
- [Matias Turkulainen](https://maturk.github.io/) (Aalto University): Core developer.
- [Ruilong Li](https://www.liruilong.cn/) (UC Berkeley): Core developer. v1.0 lead.
- [Justin Kerr](https://kerrj.github.io/) (UC Berkeley): Core developer.
- [Brent Yi](https://github.com/brentyi) (UC Berkeley): Core developer.
- [Zhuoyang Pan](https://panzhy.com/) (ShanghaiTech University): Core developer.
- [Jianbo Ye](http://www.jianboye.org/) (Amazon): Core developer.

We also have made the mathematical supplement, with conventions and derivations, available [here](https://arxiv.org/abs/2312.02121). If you find this library useful in your projects or papers, please consider citing:

```
@misc{ye2023mathematical,
    title={Mathematical Supplement for the $\texttt{gsplat}$ Library},
    author={Vickie Ye and Angjoo Kanazawa},
    year={2023},
    eprint={2312.02121},
    archivePrefix={arXiv},
    primaryClass={cs.MS}
}
```

We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [docs/DEV.md](docs/DEV.md) for more info about development.
