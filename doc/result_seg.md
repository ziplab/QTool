

## Instance Segmentation

For training and inference instructions, refer [detectron2.md](./detectron2.md).  

We provide pretrained models gradually in [google drive](https://drive.google.com/drive/folders/1vwxth9UB8AMbYP7cJxaWE9S0z9fueZ5J?usp=sharing)

Dataset | Task Method | Quantization method | Model | A/W | Reported | BBox AP / Seg AP  | Flags | Model / Log 
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:
COCO | Blenmask | LSQ | Torch-18 | 32/32 | - | 32.3/29.1 | 1x,550-R-18-Full-BN | [Link](https://drive.google.com/drive/u/1/folders/1v7Wi2QZd1cPjGir81bSDDtpIP-1TAnWv)
COCO | Blenmask | LSQ | Torch-18 | 2/2 | - | 27.5/25.2 | 1x,550-R-18-Full-BN, Quantize-Detector, doube-init | [Link](https://drive.google.com/drive/u/1/folders/1YoGuGA7QSza4FANfFPsnIMc-LQMJOV1u)
COCO | Blenmask | LSQ | Torch-18 | 2/2 | - | 25.3/23.0 | 1x,550-R-18-Full-BN, Quantize-All, doube-init | [Link](https://drive.google.com/drive/u/1/folders/1QEyumZSI7GrjorHJddu3XNfrkgk8aaTC)


Official performance of Blendmask: [Link](https://github.com/blueardour/uofa-AdelaiDet/blob/quantization/configs/BlendMask/README.md)
