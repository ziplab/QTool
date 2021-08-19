
from .classification import cifar10_loader, cifar100_loader, imagenet_loader, data_prefetcher
from .dali import dali_loader
from .fake import fake_loader

def data_loader(name):
    return {
            "cifar10": cifar10_loader,
            "cifar100": cifar100_loader,
            "imagenet": imagenet_loader,
            "dali": dali_loader,
            "fake": fake_loader,
            "tiny_imagenet": None,
           }[name]

