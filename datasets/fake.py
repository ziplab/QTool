
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import numpy as np
import PIL.Image as Image

class FakeData(torch.utils.data.Dataset):
    def __init__(self, shape, length=100000, transform=None):
        self.shape = shape
        self.transform = transform
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data = np.random.random(self.shape)
        img = Image.fromarray(data.astype('uint8')).convert('RGB')
        if self.transform is not None:
            data = self.transform(img)
        else:
            data = torch.from_numpy(data)
        label = index % 1000
        return data, label


def fake_loader(split, args=None, cfg=None):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if split == 'val':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            normalize,
            ])
        bs = args.val_batch_size
    elif split == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ])
        bs = args.batch_size
    else:
        dataset = None

    dataset = FakeData(shape=(300, 300, 3), transform=transform)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=bs, shuffle=split=='train', num_workers=args.workers, pin_memory=True)
    return loader

