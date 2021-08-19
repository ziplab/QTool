# Quantization for various computer vision tasks

The framework is able to provide quantization support for all kinds of tasks that the `Detectron2` and `AdelaiDet` projects integrate. Mix precision training is also available as a benefit.

## Install

1. install dependent packages according to [classification.md](./classification.md)

2. download the [Quantization version of detectron2](https://github.com/blueardour/detectron2) project. See [what is modified below](./detectron2.md#what-is-modified-in-the-detectron2-project).

   ```
   export FASTDIR=/workspace # change the FASTDIR as perfered
   cd $FASTDIR/git/
   git clone https://github.com/blueardour/detectron2
   # checkout the quantization branch
   cd detectron2
   git checkout quantization
   
   # install 
   pip install -e .
   
   ### other install options
   ## (add --user if you don't have permission)
   #
   ## Or if you are on macOS
   #CC=clang CXX=clang++ python -m pip install ......
   
   
   # link classification pretrained weight
   ln -s ../model-quantization/weights .
   ```
   Facebook detectron2 does not support some works such as `FCOS` and `Blendmask`. Try the [quantization version of aim-uofa/AdelaiDet](https://github.com/blueardour/AdelaiDet) for more tasks.
   
   ```
   cd $FASTDIR/git/
   git clone https://github.com/blueardour/AdelaiDet AdelaiDet
   # notice to change to the quantization branch
   cd AdelaiDet
   git checkout quantization
   
   # install
   python setup.py build develop
   
   # link classification pretrained weight
   ln -s ../model-quantization/weights .
   ```
   
   [Quantization version of detectron2](https://github.com/blueardour/detectron2) and [quantization version of AdelaiDet](https://github.com/blueardour/AdelaiDet) only add quantization support to the projects and do not change the original code logic. Quantization version projects will upgrade from their official repositories, regularly.

3. make sure the symbolic link is correct.
   ```
   cd $FASTDIR/git/detectron2
   ls -l third_party
   # the third_party/quantization should point to $FASTDIR/git/model-quantization/models
   ```

## Dataset

   Refer detectron2 datasets: [datasets/README.md](https://github.com/facebookresearch/detectron2/blob/master/datasets/README.md) and specific datasets from [AdelaiDet](https://github.com/aim-uofa/AdelaiDet).

## Pretrained models and quantization results

Configration to finetune the quantization models and pretrained weights will be gradually released.

- [Detection](./result_det.md)

- [Segmentation](./result_seg.md)

We provide pretrained models in [google drive](https://drive.google.com/drive/folders/1vwxth9UB8AMbYP7cJxaWE9S0z9fueZ5J?usp=sharing)

## What is modified in the detectron2 project

   The [`model-quantization`](https://github.com/blueardour/model-quantization) project can be used as a plugin to other projects to provide the quantization support. We modify the following files to integrate the `model-quantization` project into the `detectron2` / `AdelaiDet` projects. Use `vimdiff` to check the difference. The `model-quantization` project is potential to be equipped into other projects in a similar way.
   
   ```
   modified:   detectron2/checkpoint/detection_checkpoint.py
   modified:   detectron2/config/defaults.py
   modified:   detectron2/engine/defaults.py
   modified:   detectron2/engine/train_loop.py
   modified:   detectron2/layers/csrc/ROIAlign/ROIAlign_cuda.cu
   modified:   detectron2/layers/roi_align.py
   modified:   detectron2/layers/wrappers.py
   modified:   detectron2/modeling/backbone/fpn.py
   modified:   detectron2/modeling/meta_arch/build.py
   modified:   detectron2/modeling/meta_arch/retinanet.py
   new file:   third_party/convert_to_quantization.py
   new file:   third_party/quantization
   new file:   weights
   ```
   Make sure the `weights` and `third_party/quantization` link to correct position. 
   
   Highly recommend to check the `detectron2/engine/defaults.py` to see which options are added for the low-bit quantization.
   
   ```
   git difftool quantization master detectron2/config/defaults.py
   ```

## Known Issues

   See [know issues](./known-issues.md)
   
## Training and Testing

  Training and testing methods follow original projects ( [detectron2](https://github.com/facebookresearch/detectron2) or [aim-uofa/AdelaiDet](https://github.com/aim-uofa/AdelaiDet) ).
  
  To obtain the quantization version of the given models, please modify corresponding configuration files by setting quantization related options introduced in the quantization versions of projects. Example of the configurations for quantization are provided in `detectron2/config` and `AdelaiDet/config`, respectively. To learn how the newly introduced options impact the quantization procedure, refer option introduction in [classification.md](./classification.md#Training-script-options) for more detail explanation. We also give an sugggested instruction for the model quanzation, see below [guide](./detectron2.md#special-guide-for-quantization) and [examples](./detectron2.md#Examples) for demonstration.

<del> 

## Special guide for quantization

  The overall flow of the quantization on detection/ segmentation / text spotting tasks are as follows, some of them can be omitted if the pretrained model already exists.

- Train the full-precision backbone on Imagenet

  Refer the saved model as `backbone_full.pt`

- Finetune the low-bit backbone network

  Refer [classification.md](./classification.md) for finetuning with `backbone_full.pt` as initialization.
  
  Refer the saved model as `backbone_low.pt`
  
- Import `backbone_full.pt` and `backbone_low.pt` into detectron2 project format. 

  To import the pretrained models in correct format, refer the `renaming function` provided in `tools.py` demonstrated in [tools.md](./tools.md) and also the [examples](./detectron2.md#Examples).

- Train the full precision model with formatted `backbone_full.pt` as initialization.
  
  Refer the saved model as `overall_full.pt`
 
- Finetune the low-bit model with double-pass initialization (`overall_full.pt` and `backbone_low.pt`) or single pass initialization (`overall_full.pt`).

- Double-pass initialization is latter found no benefit.

</del>

## Examples

### Detection
  
- ResNet18-FCOS 2-bit Quantization with LSQ

<del>

  - Pretrain the full-precision and 2-bit backbone in the [`model-quantization`](https://github.com/blueardour/model-quantization) project. We provide pretrained models in [above  download links](./detectron2.md#Pretrained-models-and-quantization-results). Prepare your own model if other backbones are required. For ResNet-18, the pretrained model can be found in folder: a. Full precision model: `weights/pytorch-resnet18/resnet18_w32a32.pth`. b. 2-bit LSQ model: `weights/pytorch-resnet18/lsq_best_model_a2w2.pth`

 </del>

  - [optional] Import custom pretrained backbone

    If custom backbone is pretrained on imagenet classification. Import the weight parameters to detection project. For example, if taking the pytorch ResNet (rather than MSRA ResNet) as the backbone, run script:
  
    ```
    cd /workspace/git/model-quantization
    # prepare the weights/det-resnet18/mf.txt and weights/det-resnet18/mt.txt
    # the two files are created manually with the parameter renaming
    python tools.py --keyword update,raw --mf weights/det-resnet18/mf.txt --mt weights/det-resnet18/mt.txt --old weights/pytorch-resnet18/resnet18-5c106cde.pth --new weights/det-resnet18/resnet18_w32a32.pth
    ```
    
    The `mf.txt` and `mt.txt` files for the Resnet18 are uploaded in the `model-quantization` project as an example. The files for Resnet50 are also provided. Refer [tools.md](./tools.md) for more instructions.

  - Train full-precision FCOS-R18-1x

    There are some minor revisions on the architecture compared to official ones, such as additional ReLU layers. Check the configuration file `configs/FCOS-Detection/R_18_1x-Full-SyncBN-FixFPN.yaml`
  
    ```
    cd /workspace/git/AdelaiDet
    # add other options, such as the GPU number as needed
    python tools/train_net.py --config-file configs/FCOS-Detection/R_18_1x-Full-SyncBN-FixFPN.yaml
    ```
    
    ***Check the parameters in the backbone are re-loaded correctly***

    This step would obtain the pretrained model in `output/fcos/R_18_1x-Full-SyncBN-FixFPN/model_final.pth`

  - Fintune to get quantized model

    Check the configuration file `configs/FCOS-Detection/R_18_1x-Full-SyncBN-FixFPN-FixPoint-lsq-M2F8L8.yaml`
  
    ```
    cd /workspace/git/AdelaiDet
    # add other options, such as the GPU number as needed
    python tools/train_net.py --config configs/FCOS-Detection/R_18_1x-Full-SyncBN-FixFPN-FixPoint-lsq-M2F8L8.yaml
    ```
    
    Note: We had a bug on the config, `R_18_1x-Full-SyncBN-FixFPN-FixPoint-lsq-M2F8L8.yaml` actually set all the layers (feature maps and weight) except the input image to 2bit. The first and last layers are also 2-bit quantized, rather than 8bit quantized. Check the log file for exact configuration.
    
## License and contribution 

See [README.md](../README.md)
