

Compared with the Xnor-Net, one of the contributions of the Bi-Real net is the layer-wise full precision skip connection.

Triangle gradient instead of the STE also improves the performance.

Refer the `orign` and `xx_grad_type` options in [classification.md](./classification.md#Training-script-options)

Add `gamma` for extra learnable magnitude scale.

Notice the `padding_after_quant` option.

Also refer to [xnor-net.md](./xnor-net.md)
