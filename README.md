# Reliability-Aware Adaptive Gaussian Propagation for Low-Texture

## Overview

**RAGP** is a reliability-aware adaptive Gaussian propagation framework for reconstructing static regions in **low-texture dynamic outdoor scenes**.

It introduces two key components:

- **Dynamic-aware static Gaussian reliability optimization**, which combines dynamic priors and rendering residuals to guide static supervision, propagation filtering, and Gaussian densification.
- **Static-reliability-guided adaptive patch-based geometric propagation**, which selects suitable patch scales according to local structure, geometric consistency, and static reliability.

RAGP improves propagation stability in low-texture regions while reducing erroneous propagation and densification around dynamic interference and structural boundaries.

<p align="center">
  <img src="assets/framework.png" width="90%">
</p>

## Updates

- **2026/08:** Part of the implementation has been released. The remaining code will be released in future updates.
  
## Installation

Follow the instructions in the [GaussianPro](https://github.com/kcheng1021/GaussianPro)  to setup the environment. 

## Datasets

We conduct experiments on selected scenes from the following datasets:

- [NeRF On-the-go](https://rwn17.github.io/nerf-on-the-go/), which contains casually captured real-world scenes with dynamic distractors.
- [nuScenes](https://www.nuscenes.org/), which provides large-scale urban driving scenes with complex dynamic environments.

Please refer to the official websites for dataset access and download instructions.

## Acknowledgements

We thank the following projects:

- [GaussianPro](https://github.com/kcheng1021/GaussianPro)
- [desplat](https://github.com/AaltoML/desplat)
