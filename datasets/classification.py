
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
from PIL import Image, ImageFile

class Lighting(object):
    """Lighting noise(AlexNet - style PCA - based noise)"""
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        # img is supposed go be a torch tensor

        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))

__imagenet_pca = {
    'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
    'eigvec': torch.Tensor([
        [-0.5675,  0.7192,  0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948,  0.4203],
    ])
}

def fast_collate(batch):
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8 )
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)

        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets

# for issue similar with https://github.com/python-pillow/Pillow/issues/1510
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except IOError as e:
        log_file = 'debug-image-%d.txt' % os.getpid()
        if os.path.isfile(log_file):
            log_file = open(log_file, 'a')
        else:
            log_file = open(log_file, 'w')
        print("read image %s. Error: " % path, e, file=log_file)
        log_file.close()
        # fake image
        raise RuntimeError("failed to read file %r" % path)
        img = Image.new('RGB', (256, 256))
        return img

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def fix_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def imagenet_loader(split, args=None, cfg=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.distributed:
        if split == 'val':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(args.input_size),
                #transforms.ToTensor(),
                #normalize,
                ])
            bs = args.val_batch_size
        elif split == 'train':
            transform = transforms.Compose([
                transforms.RandomResizedCrop(args.input_size),
                transforms.RandomHorizontalFlip(),
                #transforms.ToTensor(),
                #normalize,
                ])
            bs = args.batch_size
        else:
            dataset = None
    else:
        if split == 'val':
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                normalize,
                ])
            bs = args.val_batch_size
        elif split == 'train':
            if args.addition_augment:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(.4,.4,.4),
                    transforms.ToTensor(),
                    Lighting(0.1, __imagenet_pca['eigval'], __imagenet_pca['eigvec']),
                    normalize,
                    ])
            else:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(args.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                    ])
            bs = args.batch_size
        else:
            dataset = None

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    dataset = datasets.folder.ImageFolder(root=os.path.join(args.root, split), transform=transform, loader=fix_loader)
    sampler = None
    collate_fn = None
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        collate_fn = fast_collate

    shuffle = sampler is None if args.distributed else split=='train'
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=bs, shuffle=shuffle, num_workers=args.workers, pin_memory=(args.device_ids is not None),
            sampler=sampler, collate_fn=collate_fn)
    return loader

def cifar_loader(split, args=None, cfg=None, torch_dataset=None):
    #normalize = transforms.Normalize(mean=[0.507, 0.4865, 0.441], std=[0.267, 0.256, 0.276])
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    if split == "val":
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            ])
        bs = args.val_batch_size
    elif split == 'train':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Pad(padding=4, padding_mode='reflect'),
            transforms.RandomCrop(32, padding=0),
            #transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
            ])
        bs = args.batch_size
    else:
        dataset = None

    dataset = torch_dataset(root=args.root, train=split=='train', download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=bs, shuffle=split=='train', num_workers=args.workers, pin_memory=True)
    return loader

def cifar10_loader(split, args=None, cfg=None):
    return cifar_loader(split, args, cfg, torchvision.datasets.CIFAR10)

def cifar100_loader(split, args=None, cfg=None):
    return cifar_loader(split, args, cfg, torchvision.datasets.CIFAR100)

class data_prefetcher():
    def __init__(self, loader, transform=True):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        self.transform = transform
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_target = self.next_target.long()
            if not self.transform:
                return
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        if self.transform:
            torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None and self.transform:
            input.record_stream(torch.cuda.current_stream())
        if target is not None and self.transform:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

