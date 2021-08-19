
# QTool: A Low-bit Quantization Toolbox for Deep Neural Networks in Computer Vision

This project provides abundant choices of quantization strategies (such as the quantization algorithms, training schedules and empirical tricks) for quantizing the deep neural networks into low-bit counterparts. This project can act as a flexible plugin and benefit various computer vision tasks, such as image classification, dense detection and segmentation, text parsing and super resolution. Pretrained models are provided to show high standard of the code on achieving appealing quantization performance. 

## Instructions for different tasks

- [Classification](./doc/classification.md)
- [Detection / Segmentation / Text parsing ](./doc/detectron2.md)
- [Super Resolution](./doc/edsr.md)

## Update History

- 2020.12.12 Text parsing
- 2020.11.01 Super Resolution
- 2020.07.08 Instance Segmentation
- 2020.07.08 Object Detection
- 2020.06.23 Add classification quantization

## Citation

Please cite the following work if you find the project helpful.

```
@misc{chen2020qtool,
author = {Peng Chen, Bohan Zhuang, Jing Liu and Chunlei Liu},
title = {{QTool: A Low-bit Quantization Toolbox for Deep Neural Networks in Computer Vision}},
year = {2020},
howpublished = {\url{https://github.com/MonashAI/QTool/}},
note = {Accessed: [Insert date here]}
}
```

This project includes the implementation of some of our works: 
```
@inproceedings{chen2021aqd,
  title={Aqd: Towards accurate quantized object detection},
  author={Chen, Peng and Liu, Jing and Zhuang, Bohan and Tan, Mingkui and Shen, Chunhua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={104--113},
  year={2021}
}

@inproceedings{chen2021fatnn,
  title={FATNN: Fast and Accurate Ternary Neural Networks},
  author={Chen, Peng and Zhuang, Bohan and Shen, Chunhua},
  booktitle={Proceedings of the International Conference on Computer Vision},
  year={2021}
}

```

**Also cite the corresponding publications when you choose [dedicated algorithms](./doc/reference.md).**

We are integrating more of our work and other great studies into this project. 

## Contribute

To contribute, PR is appreciated and suggestions are welcome to discuss with.

## License

For academic use, this project is licensed under the 2-clause BSD License. See LICENSE file. For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com) and [Peng Chen](mailto:blueardour@gmail.com).

