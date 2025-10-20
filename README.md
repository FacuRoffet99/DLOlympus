# DLOlympus 🏛️

DLOlympus is a toolkit to accelerate and simplify Deep Learning workflows. This repository provides a set of reusable classes and functions designed to solve common problems in classification, regression, and object detection tasks. Start your journey by looking at the `pipelines_nbs` folder, which contains notebooks with complete workflows for training and inference.


## Olympians: Supported Functionality

`DLOlympus` offers specific features to enhance your favorite frameworks:

* **⚡ fastai ⚡** - The all-powerfull king of classification and regression
    * Multi-head model builders.
    * Improved metrics.
    * Custom DataBlocks and flexible loss functions.
    * Support for `albumentations` transforms.
    * Simple and robust inference.
    * Class imbalance samplers and loss weighting.
    * Export to ONNX format.
* **🔱 MMDetection 🔱** - The ruler of object detection and instance segmentation (*In progress*)
    * Validation loss support.
    * Visualization of samples.
    * Standardizer inferencer.
    * Plots for losses and metrics.
* **⚕️ ONNX ⚕️** - The communicator of frameworks:
    * Inferencer for easy predictions.


## Installation

First, install `torch` using the official documentation. Then, to install DLOlympus and its dependecies, run the following

```bash
git clone https://github.com/FacuRoffet99/DLOlympus.git
cd DLOlympus
pip install -e .
```

If you plan to use MMDetection functionality, you must install it first. You may follow the [official MMDetection installation instructions](https://mmdetection.readthedocs.io/en/latest/get_started.html) or my (almost) headache-free guide in `install_mmdetection.ipynb`.


## Contributions to Open Source

Work from this repository has led to the following contributions:
* Fixes:
    * Default arguments in SaveModelCallback [fastai#4118](https://github.com/fastai/fastai/pull/4118)
    * Deprecation warning in MixedPrecision [fastai#4124](https://github.com/fastai/fastai/pull/4124)