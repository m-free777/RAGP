# Reliability-Aware Adaptive Gaussian Propagation for Low-Texture

## Overview

RAGP is a method for robust 3D Gaussian Splatting in dynamic scenes.

Our method focuses on improving static scene reconstruction in the presence of dynamic objects by introducing:

- Dynamic-aware static Gaussian reliability optimization
- Static-reliability-guided adaptive patch-based geometric propagation

<p align="center">
  <img src="assets/framework.png" width="90%">
</p>

## Installation

Follow the instructions in the [GaussianPro](https://github.com/kcheng1021/GaussianPro)  to setup the environment. 

## Datasets

We conduct experiments on selected scenes from the following datasets:

- [NeRF On-the-go](https://rwn17.github.io/nerf-on-the-go/), which contains casually captured real-world scenes with dynamic distractors.
- [nuScenes](https://www.nuscenes.org/), which provides large-scale urban driving scenes with complex dynamic environments.

Please refer to the official websites for dataset access and download instructions.

## Code Release

We have released part of the implementation. The remaining code will be made available in a future update.

## Acknowledgements

We thank the following projects:

- [GaussianPro](https://github.com/kcheng1021/GaussianPro)
- [desplat](https://github.com/AaltoML/desplat)
