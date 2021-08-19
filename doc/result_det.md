
Here lists selected experiment result. The performance is potentially being better if more effort is paid on tuning. See [experience.md](experience.md) to communicate training skills.

### Detection

For training and inference instructions, refer [detectron2.md](./detectron2.md). 
As the project is keeping upgrading, the pretrained model provided on [Google Drive](./detectron2.md#Pretrained-model) might show better performance compared with the one in table.
For more details, please refer to [our paper](https://arxiv.org/abs/2007.06919).

Dataset | Task Method | Quantization method | Model | A/W | Reported | AP  | Flags 
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:
COCO | Retina-Net | - | Torch-18 | 32/32 | - | 31.5 | 1x
COCO | Retina-Net | - | Torch-18 | 32/32 | - | 32.8 | 1x, FPN-BN,Head-GN
COCO | Retina-Net | - | Torch-18 | 32/32 | - | 33.0 | 1x, FPN-BN,Head-BN
COCO | Retina-Net | - | Torch-34 | 32/32 | - | 35.2 | 1x
COCO | Retina-Net | - | Torch-50 | 32/32 | - | 36.6 | 1x
COCO | Retina-Net | - | Torch-50 | 32/32 | - | 37.8 | 1x, FPN-BN,Head-BN
COCO | Retina-Net | - | MSRA-R50 | 32/32 | - | 36.4 | 1x
COCO | Retina-Net | - | Torch-18 | 4/4 | - | 34.0 | 1x,Full-BN, Quantize-All
COCO | Retina-Net | - | Torch-18 | 3/3 | - | 32.8 | 1x,Full-BN, Quantize-All
COCO | Retina-Net | - | Torch-18 | 2/2 | - | 29.6 | 1x,Full-BN, Quantize-All
COCO | Retina-Net | - | Torch-34 | 4/4 | - | 37.0 | 1x,Full-BN, Quantize-All
COCO | Retina-Net | - | Torch-34 | 3/3 | - | 35.9 | 1x,Full-BN, Quantize-All
COCO | Retina-Net | - | Torch-34 | 2/2 | - | 33.1 | 1x,Full-BN, Quantize-All
COCO | FCOS | - | MSRA-R50 | 32/32 | - | 38.6 | 1x
COCO | FCOS | - | Torch-50 | 32/32 | - | 38.4 | 1x
COCO | FCOS | - | Torch-50 | 32/32 | - | 38.5 | 1x,FPN-BN
COCO | FCOS | - | Torch-50 | 32/32 | - | 38.9 | 1x,FPN-BN,Head-BN
COCO | FCOS | - | Torch-34 | 32/32 | - | 37.3 | 1x
COCO | FCOS | - | Torch-18 | 32/32 | - | 32.2 | 1x
COCO | FCOS | - | Torch-18 | 32/32 | - | 33.4 | 1x,FPN-BN
COCO | FCOS | - | Torch-18 | 32/32 | - | 33.9 | 1x,FPN-BN, FP16
COCO | FCOS | - | Torch-18 | 32/32 | - | 33.9 | 1x,FPN-BN,Head-BN
COCO | FCOS | - | Torch-18 | 32/32 | - | 34.3 | 1x,FPN-SyncBN,Head-SyncBN
COCO | FCOS | - | Torch-18 | 4/4 | - | 35.2 | 1x,FPN-BN, Quantize-All, double-init
COCO | FCOS | - | Torch-18 | 3/3 | - | 34.1 | 1x,FPN-BN, Quantize-All, double-init
COCO | FCOS | - | Torch-18 | 2/2 | - | 33.4 | 1x,FPN-BN, Quantize-Backbone, double-init
COCO | FCOS | - | Torch-18 | 2/2 | - | 32.0 | 1x,FPN-BN, Quantize-All, singe-pass-init
COCO | FCOS | - | Torch-18 | 2/2 | - | 30.3 | 1x,FPN-BN, Quantize-All, double-init
COCO | FCOS | LQ-Net | Torch-18 | ter/ter | - | 32.6 | 1x,FPN-BN, Quantize-Backbone, double-init
COCO | FCOS | LQ-Net | Torch-18 | ter/ter | - | 26.2 | 1x,FPN-BN, Quantize-All, double-init

Flags:

`FPN-BN` indicates adding BN and RELU in the FPN; `FP16` implies the case is trained in FP16 (half float) mode; `Head-BN` represents the prospoal header employes non shared BatchNorm. `Full-BN` indicates combining `FPN-BN` and `Head-BN`. `Torch-18/34/50` means the backbone is the Pytorch ResNet-18/34/50.

