from setuptools import setup, find_packages

setup(
    name="sensaug",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[],  # or ['numpy', 'pandas'], etc.
    author="Laura Zheng",
    description="PyTorch implementation for Adaptive Sensitivity Analysis for Robust Augmentation against Natural Corruptions in Image Segmentation.",
)
