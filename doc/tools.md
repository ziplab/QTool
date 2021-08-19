
# A independent script for debug / model converting

This page presents selected functions provided by `tools.py` which is commonly used in the project.

## Model import and paramter renaming.

Some times models are trained by the another (or old version) repo. When importing the pretrained model to new repo, the name of certain variables /paramters might be changed.

The following commands can be used to browse items in the pretrained model and renaming specific parameters (***Remove the brackets when testing, brackets only indicates the enclosed one can be replaced with other string***).

1. looking up item in the model file.

```
# cd weights/pytorch-resnet50/
# download pretrained model by
# wget https://download.pytorch.org/models/resnet50-19c8e357.pth
python tools.py --keyword verbose --verbose_list all --old [weights/pytorch-resnet50/resnet50-19c8e357.pth]

# save to name into text
python tools.py --keyword verbose --verbose_list all --old [weights/pytorch-resnet50/resnet50-19c8e357.pth] | awk '{ print $1}' | tee name_list.txt
```

2. renaming parameter

Export pytorch official resnet model to Detectron2 format as initialization model. Edit your own `mapping_from.txt` and `mapping_from.txt` file based on the naming space (which can be browsed by above command)
```
python tools.py --keyword update[,raw]  --mf [weights/det-resnet50/mapping_from.txt] --mt [weights/det-resnet50/mapping_to.txt] --old [weights/pytorch-resnet50/resnet50-19c8e357.pth] --new [weights/det-resnet50/official-r50.pth]
```

Add `raw` in the keyword to generate the model file with/without `state_dict` key.

Another example for LSQ 2bit model converting:
```
 python ../model-quantization/tools.py --keyword update,raw --mf weights/det-resnet50/mf_lsq_2bit.txt --mt weights/det-resnet50/mt_lsq_2bit.txt --old weights/pytorch-resnet50/lsq_best_model_a2w2-new.pth --new weights/det-resnet50/lsq_best_model_a2w2.pth
```
