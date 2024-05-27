# **NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces by Enforcing Zero Gaussian Curvature**

### [Project](https://qiujiedong.github.io/publications/NeurCADRecon/) | [Paper](https://arxiv.org/pdf/2404.13420.pdf)

**This code is based on the IGR, we also provide the implementation based on the SIREN: [NeurCADRecon](https://github.com/QiujieDong/NeurCADRecon)**

## Requirements

- python 3.7
- CUDA 12.2
- pytorch 1.13.0

## Installation

```
git clone https://github.com/QiujieDong/NeurCADRecon_IGR.git
cd NeurCADRecon_IGR
```

## Preprocessing

Referring to [NeurCADRecon](https://github.com/QiujieDong/NeurCADRecon).

## Overfitting

```angular2html
cd ./code/reconstruction
python run.py
```

## Cite

If you find our work useful for your research, please consider citing the following papers :)

```bibtex
@article{Dong2024NeurCADRecon,
author={Dong, Qiujie and Xu, Rui and Wang, Pengfei and Chen, Shuangmin and Xin, Shiqing and Jia, Xiaohong and Wang, Wenping and Tu, Changhe},
title={NeurCADRecon: Neural Representation for Reconstructing CAD Surfaces by Enforcing Zero Gaussian Curvature},
journal={ACM Transactions on Graphics},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
year={2024},
month={July},
volume = {43},
number={4},
doi={10.1145/3658171},
keywords = {CAD model, unoriented point cloud, surface reconstruction, signed distance function, Gaussian curvature}
}
```


## Acknowledgments
Our code is inspired by [Neural-Singular-Hessian](https://github.com/bearprin/Neural-Singular-Hessian),  [SIREN](https://github.com/vsitzmann/siren), and [IGR](https://github.com/amosgropp/IGR).

