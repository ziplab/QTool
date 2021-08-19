
# Quantization for Super Resolution (EDSR)

## Install


1. install quantization dependent packages according to [classification.md](./classification.md)

   Also, refer the [original dependencies](https://github.com/blueardour/EDSR-PyTorch#dependencies) for the SR task dependencies.

2. download the [quantization version of EDSR-PyTorch](https://github.com/blueardour/EDSR-PyTorch) project.

   ```
   cd /workspace/git/
   git clone https://github.com/blueardour/EDSR-PyTorch
   # checkout the quantization branch
   cd EDSR-PyTorch
   git checkout quantization
 
   ```

3. make sure the symbolic link is correct. Create the link if not exists.
   ```
   cd /workspace/git/EDSR-PyTorch/src/
   ls -l third_party
   # the third_party/quantization should point to /workspace/git/model-quantization
   ```


## Dataset / Train / Test

   Refer instruction from EDSR-PyTorch: [how-to-train-edsr-and-mdsr](https://github.com/blueardour/EDSR-PyTorch#how-to-train-edsr-and-mdsr)
   
   ***Link your training data to /workspace/git/EDSR-PyTorch/data*** The `train.sh` script in `src` folder finds data in `../data` folder by default.
   
## To start

```
cd /workspace/git/EDSR-PyTorch/src
bash train.sh config.*
```

Ongoing `config.*` is advised to put in `/workspace/git/EDSR-PyTorch/src`. Verified files which are to be released could be moved to `/workspace/git/EDSR-PyTorch/src/config`.

## Pretrained models and quantization results

- [Super resolution](./result_sr.md)

We provide pretrained models gradually in [google drive](https://drive.google.com/drive/folders/1vwxth9UB8AMbYP7cJxaWE9S0z9fueZ5J?usp=sharing)
