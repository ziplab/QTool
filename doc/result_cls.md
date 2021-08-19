
### Classification

Refer [classification.md](./classification.md) for detailed instructions.

Both the Top-1(\%) from original paper and the reproduction are listed. Corresponding training and testing configurations can be found in the `config` folder. Selected experiment results are listed. Users are encouraged to try different configurations to implement their own targets.

Note that the performance among different methods is obtained based on different training hyper-parameters. The accuracy in the table will not be the evidence of superior of one algorithm over another. Training hyper-parameters and tricks (such as `weight normalization`) play a considerable role on improving the performance. See the summary of my experience on training quantization networks in [experience.md](./experience.md).

We provide pretrained models in [google drive](https://drive.google.com/drive/folders/1vwxth9UB8AMbYP7cJxaWE9S0z9fueZ5J?usp=sharing). Report missing pretrained model files if you cannot find it. 

Try the 67.8\% Top-1 LSQ quantization for ResNet-18 `bash train.sh config/config.lsq.eval.imagenet.2bit.resnet18`.


Dataset | Method | Model | A/W | Reported | Top-1  | Flags | Config
--- |:---:|:---:|:---:|:---:|:---:|:---:|:---:
imagenet | - | ResNet-18 | 32/32 | - | 70.1 | PreBN,bacs | [File](../config/config.dorefa.eval.imagenet.fp.resnet18)
imagenet | - | Torch-R18 | 32/32 | 69.8 | 70.1 | Pytorch-official |[File](../config/config.dorefa.eval.imagenet.fp.torch-resnet18)
imagenet | Fixup | ResNet-18 | 32/32 | - | 69.0 | fixup,cbsa,mixup=0.7 | [File](../config/)
imagenet | Fixup | ResNet-50 | 32/32 | - | 75.9 | fixup,cbsa,mixup=0.7 | [File](../config/config.fixup.eval.imagenet.fp.resnet50)
imagenet | TResnet | ResNet-18 | 32/32 | 70.1 | 68.7 | PreBN,bacs,TResNetStem | [File](../config/config.TResNet.eval.dali.fp.resnet18)
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 64.9 | PreBN,bacs, ep120 (old)  | [File](../config/config.lq-net.eval.dali.2bit.resnet18)
imagenet | LQ-net | ResNet-18 | 2/2 | - | 65.9 | PreBN,bacs,fm_qg=8, ep120 (old) | [File](../config/config.lq-net.eval.dali.2bit.resnet18-fg8)
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.7 | PreBN,bacs, ep120 | [File](../config/config.lq-net.finetune.dali.2bit.resnet18-baseline-sgdr)
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.3 | PreBN,bacs,wt_mean-var, ep40
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.6 | PreBN,bacs,wt_mean-var, ep120 | [File](../config/config.lq-net.finetune.dali.2bit.resnet18-wt-norm)
imagenet | LQ-net | ResNet-18 | 2/2 | 64.9 | 65.4 | PreBN,bacs,wt_mean-var,wt_gq=1, ep120
imagenet | LSQ | Torch-R18 | 2/2 | 67.6 | 67.3 | vanilla resnet(paper use pre act) | [File](../config/config.lsq.eval.imagenet.2bit.torch-resnet18)
imagenet | Dorefa-Net | ResNet-18 | 2/2 | - | 64.1 | PreBN,bacs  | [File](../config/config.dorefa.eval.imagenet.2bit.resnet18)
imagenet | Group-Net | ResNet-18 | 1/1 | - | 63.9 | cabs,bireal,base=5,without-softgate | [File](../config/config.group-net.eval.imagenet.bin.resnet18.base5.cabs)
imagenet | Xnor-Net | ResNet-18 | 1/1 | 51.2 | 52.0 | cbsa,fm_triangle,wt_pass,No-ReLU | [File](../config/config.xnor.train-scratch.dali.bin.resnet18-triangle-pass)
imagenet | Xnor-Net | ResNet-18 | 1/1 | 51.2 | 50.5 | cbsa,fm_STE,wt_pass,No-ReLU
imagenet | LSQ | Torch-R18 | 1/1 | - | 58.5 | ReLU,wt-var-mean,wtg=1
imagenet | LSQ | Torch-R18 | t/t | - | 65.1 | wd2.5e-5,wt_qg=1_var-mean,ns,ds,sgd_0,fp32,ep90
imagenet | LSQ | Torch-R34 | t/t | - | 69.2 | wd2.5e-5,wt_qg=1_var-mean,ns,ds,sgd_0,fp32,ep90
imagenet | LSQ | Torch-R50 | t/t | - | 72.6 | wd2.5e-5,wt_qg=1_var-mean,ns,ds,sgd_0,fp32,ep90
imagenet | LSQ | Torch-R18 | 2/2 | - | 66.9 | wd2.5e-5,wt_qg=1_var-mean,ns,ds,sgd_0,fp32,ep90
imagenet | LSQ | ResNet-18 | 2/2 | - | 67.8 | wd2.5e-5,wt_qg=1_var-mean,sgd_1,fp32,ep90,kd | [File](../config/config.lsq.eval.imagenet.2bit.resnet18)
imagenet | non-uniform | Torch-R18 | 2/2 | - | 66.8 | wd2.5e-5,sc3.0,wt_qg=1_var-mean,ns,ds,clrd,sgd_0,fp32,ep90
imagenet | non-uniform | Torch-R18 | 2/2 | - | 65.5 | wd2e-5,sc3.0,wt_qg=1_var-mean,ns,ds,sgd_2,fp32,ep40
dali | non-uniform | Torch-R18 | 2/2 | - | 65.8 | wd2e-5,sc3.0,wt_qg=1,ns,ds,sgd_2,fp16,ep40
imagenet | non-uniform  | Torch-R18 | t/t | - | 65.0 | wd2.5e-5,wt_qg=1_var-mean,ns,ds,clrd,sgd_0,fp32,ep90
imagenet | non-uniform | Torch-R18 | t/t | - | 59.23 | wd2.5e-5,wt_qg=1_var-mean,ns,ds,clr_wd2.5e-5,sgd_0,fp32,ep90,train-scratch
imagenet | non-uniform | Torch-R18 | t/t | - | 64.8 | wd2.5e-5,wt_qg=1_var-mean,ns,ds,sgd_0,fp32,ep90
imagenet | non-uniform-D | Torch-R18 | t/t | - | 65.0 | wd2.5e-5,wt_qg=1_var-mean,ns,ds,clr_wd2.5e-5,sgd_0,fp32,ep90
imagenet | non-uniform-D | Torch-R18 | t/t | - | 64.8 | wd2.5e-5,wt_qg=1_var-mean,ns,ds,clrd,sgd_0,fp32,ep90
cifar100 |  - | ResNet-20 | 32/32 | - | 67.41 | cbsa, ldn, baseline
cifar100 |  - | ResNet-20 | 32/32 | - | 66.92 | cbsa, ldn, order c
cifar100 |  - | ResNet-20 | 32/32 | - | 67.73 | cbsa, ldn, order cb
cifar100 |  - | ResNet-20 | 32/32 | - | 66.23 | cbsa, ldn, order ca
cifar100 |  - | ResNet-20 | 32/32 | - | 68.04 | cbsa, ldn, order cba
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 63.92 | cbsa, ldq, baseline, real shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 62.74 | cbsa, ldq, order c, real shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 65.73 | cbsa, ldq, order cb, real shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 58.88 | cbsa, ldq, order ca, real shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 65.79 | cbsa, ldq, order cba, real shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 61.86 | cbsa, ldq, baseline, 2bit shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - |  1.00 | cbsa, ldq, order c, 2bit shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 63.00 | cbsa, ldq, order cb, 2bit shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - |  1.00 | cbsa, ldq, order ca, 2bit shortcut
cifar100 |  LSQ | ResNet-20 | 2/2 | - | 62.37 | cbsa, ldq, order cba, 2bit shortcut
dali | - | ResNet-18 | 32/32 | 69.8 | 70.5 | cbsa, ldn, order cb, fp16, sgd_2
imagenet |  - | ResNet-18 | 32/32 | 69.8 | 71.4 | cbsa, ldn, order cb, fp32, sgd_2
cifar100 |  - | ResNet-18 | 32/32 | - | 68.20 | cbsa,baseline
cifar100 |  - | ResNet-18 | 32/32 | - | 64.85 | cbsa,prone,npd,keepdim,postbn
cifar100 |  - | ResNet-50 | 32/32 | - | 70.26 | cbsa,baseline
cifar100 |  - | ResNet-50 | 32/32 | - | 70.18 | cbsa,prone,npd,keepdim,postbn

`Torch-Rxx` indicates the ResNet architecture from Pytorch (so-called vanilla structure). `ResNet-xx` represnets the variants of ResNet. Minior differences are observed from different implementation from other projects. We provide flexible structure control to build compatibility of those projects. See [resnet.md](./resnet.md) for the architecture description and [classification.md](./classification.md) for how to control the choice by different configuration.

Explanations on some flags:

- cbsa / bacs:
  The resnet conv seq
  
- wt_var-mean:
  apply weight normalization (type `var-mean`) on the weight
  
- ep40 / ep120:
  total epoch of 40 / 120 in the training
  
- fm_qg/ wt_qg:
  quantization group
  
- real shortcut / real-skip: the downsample layer is kept in full precision. Other wise the shortcut is quantized (eg. `2bit shortcut`)

- old:  indicating that better results are obtained but still not updated in the table.

