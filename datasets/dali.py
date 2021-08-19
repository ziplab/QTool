
import os
import torch

dali_enable = True
try:
    if torch.cuda.is_available():
        from nvidia.dali.plugin.pytorch import DALIClassificationIterator
        from nvidia.dali.pipeline import Pipeline
        import nvidia.dali.ops as ops
        import nvidia.dali.types as types
    else:
        dali_enable = False
except ImportError:
    dali_enable = False
    print("Can not import DALI. Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")
    #raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

if dali_enable:
  class HybridPipe(Pipeline):
      def __init__(self, split, args=None):
          #batch_size, num_threads, data_dir, crop, shuffle=False, device_id=0, size=256, dali_cpu=False):
          self.split = split
          self.bs = args.batch_size if self.split == 'train' else args.val_batch_size
          self.shuffle = self.split == 'train'
          self.data_dir = os.path.join(args.root, split)
          self.crop = args.input_size
          super(HybridPipe, self).__init__(self.bs, args.workers, 0, seed=12)
          self.input = ops.FileReader(file_root=self.data_dir, shard_id=0, num_shards=1, random_shuffle=self.shuffle)
          dali_device = "gpu"
          if split == 'train':
              self.decode = ops.ImageDecoderRandomCrop(device="mixed", output_type=types.RGB,
                      device_memory_padding=211025920, host_memory_padding=140544512,
                      random_aspect_ratio=[0.75, 1.333],
                      random_area=[0.08, 1.0],
                      num_attempts=100)
              #self.res = ops.Resize(device=dali_device, resize_x=args.input_size, resize_y=args.input_size, interp_type=types.INTERP_TRIANGULAR)
              self.res = ops.Resize(device=dali_device, resize_shorter=args.input_size, interp_type=types.INTERP_TRIANGULAR)
          else:
              self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
              self.res = ops.Resize(device=dali_device, resize_shorter=256, interp_type=types.INTERP_TRIANGULAR)

          self.cmnp = ops.CropMirrorNormalize(device="gpu",
                  output_dtype=types.FLOAT,
                  output_layout=types.NCHW,
                  crop=(self.crop, self.crop),
                  image_type=types.RGB,
                  mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                  std=[0.229 * 255,0.224 * 255,0.225 * 255])
          self.coin = ops.CoinFlip(probability=0.5)
  
      def define_graph(self):
          if self.split == 'train':
              rng = self.coin()
          self.jpegs, self.labels = self.input(name="Reader")
          images = self.decode(self.jpegs)
          images = self.res(images)
          if self.split == 'train':
              output = self.cmnp(images.gpu(), mirror=rng)
          else:
              output = self.cmnp(images.gpu())
          return [output, self.labels]
  
  def dali_loader(split, args=None, cfg=None):
      pipe = HybridPipe(split, args=args)
      pipe.build()
      loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))
      #print(loader.__dict__)
      #print(loader._size)
      #print(loader.size)
      return loader

else:
  def dali_loader(split, args=None, cfg=None):
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")



