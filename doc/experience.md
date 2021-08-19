
## Training tricks

- Learning rate

- Activation Function

- Initialization

  For quantization in all kinds of tasks, it is advised to first train the full precision model and then quantization by finetuning  with the full precision model as the initialization.

  This strategy shows little improvement in the BNN training for image classification task, however, poses considerable benefit for higher bit quantization. It is found to be important for detection and segmentation tasks.
  
  Another trick about the initialization is the `Initialization based on statistic on small amount of data`. It means, before the finetuning procedure, we can first take samples in the dataset and initialize the custom variables (for example, the `clip_val` in Dorefa-net based methods or the `basis` in LQ-net) based these samples. We provide the support by add the `stable`, `warmup` and `custom-update` options.


