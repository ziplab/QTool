
import logging

try:
    from .resnet_ import resnet18, resnet20_, resnet20, resnet32, resnet34, resnet34_, resnet44, resnet50, resnet101
    from .resnet  import resnet18 as pytorch_resnet18
    from .resnet  import resnet34 as pytorch_resnet34
    from .resnet  import resnet50 as pytorch_resnet50
    from .resnet  import resnet101 as pytorch_resnet101

    #from .vgg import vgg16_bn as pytorch_vgg16bn
    from .vgg_small_ import vgg_small
    
    #from .alexnet_ import alexnet as alexnet_  # revise from XNOR-Net-PyTorch
    from .nin_ import nin as nin_
    
    #from .mobilenet import mobilenetv2, mobilenetv1

    #from .densenet import densenet121
    #from .nasnet import nasnet
    #from .squeezenet import squeezenet

    from . import policy 
except (ImportError, RuntimeError, FileNotFoundError, PermissionError) as e:
    print('import classification model failed', e)

third_party = True
try:
    from third_party import model_zoo as third_party_model_zoo
    from third_party import get_model as third_party_get_model
except (ImportError, RuntimeError, FileNotFoundError, PermissionError) as e:
    print('loading third party model failed', e)
    third_party = False

model_zoo = [
  'pytorch-resnet18',
  'pytorch-resnet34',
  'pytorch-resnet50',
  'pytorch-resnet101',
  'pytorch-vgg16bn',
  'se-resnet18',
  'se-resnet50',
  'resnet18',
  'resnet20',
  'resnet20_',
  'resnet32',
  'resnet34',
  'resnet34_',
  'resnet44',
  'resnet50',
  'resnet101',
  'mobilenetv1',
  'mobilenetv2',
  'qmobilenetv1',
  'densenet121',
  'nasnet',
  'squeezenet',
  'nin',
  'alexnet',
  'vgg_small'
  ]

if third_party:
    model_zoo += third_party_model_zoo

def get_model(args):
    if third_party and args.model in third_party_model_zoo:
        return third_party_get_model(args)

    if 'cifar10' in args.keyword:
        num_classes = 10
        input_size = 32
    elif 'cifar100' in args.keyword:
        num_classes = 100
        input_size = 32
    else:
        num_classes = 1000
        input_size = 224

    if args.model == 'alexnet':
        input_size = 227

    fm_bit = wt_bit = 32
    if getattr(args, 'fm_bit', None) is not None:
        fm_bit = getattr(args, 'fm_bit')
    else:
        setattr(args, 'fm_bit', fm_bit)

    if getattr(args, 'wt_bit', None) is not None:
        wt_bit = getattr(args, 'wt_bit')
    else:
        setattr(args, 'wt_bit', wt_bit)

    if getattr(args, 'num_classes', None) is not None:
        num_classes = getattr(args, 'num_classes')
    else:
        setattr(args, 'num_classes', num_classes)

    if getattr(args, 'input_size', None) is not None:
        input_size = getattr(args, 'input_size')
    else:
        setattr(args, 'input_size', input_size)

    logging.info('update fm_bit %r', fm_bit)
    logging.info('update wt_bit %r', wt_bit)
    logging.info('update num_classes %d', num_classes)
    logging.info('update input_size %d', input_size)

    if args.model == 'resnet18':
        return resnet18(args), args
    elif args.model == 'resnet20':
        return resnet20(args), args
    elif args.model == 'resnet20_':
        return resnet20_(args), args
    elif args.model == 'resnet32':
        return resnet32(args), args
    elif args.model == 'resnet34':
        return resnet34(args), args
    elif args.model == 'resnet34_':
        return resnet34_(args), args
    elif args.model == 'resnet44':
        return resnet44(args), args
    elif args.model == 'resnet50':
        return resnet50(args), args
    elif args.model == 'resnet101':
        return resnet101(args), args

    elif args.model == 'alexnet':
        return alexnet_(args), args

    elif args.model == 'mobilenetv1':
        return mobilenetv1(args), args
    elif args.model == 'mobilenetv2':
        return mobilenetv2(args), args
    elif args.model == 'qmobilenetv1':
        return qmobilenetv1(args=args), args

    elif args.model == 'vgg_small':
        return vgg_small(args), args

    elif args.model == 'nin':
        return nin_(args=args), args

    elif args.model == 'pytorch-resnet18':
        model = pytorch_resnet18(pretrained=False, progress=True, args=args)
        return model, args
    elif args.model == 'pytorch-resnet34':
        model = pytorch_resnet34(pretrained=False, progress=True, args=args)
        return model, args
    elif args.model == 'pytorch-resnet50':
        model = pytorch_resnet50(pretrained=False, progress=True, args=args)
        return model, args
    elif args.model == 'pytorch-resnet101':
        model = pytorch_resnet101(pretrained=False, progress=True, args=args)
        return model, args
    elif args.model == 'pytorch-vgg16bn':
        model = pytorch_vgg16bn(pretrained=False, progress=True, args=args)
        return model, args

    elif args.model == 'se-resnet18':
        model = se_resnet18()
        return model, args
    elif args.model == 'se-resnet50':
        model = se_resnet50()
        return model, args
    else:
        return None, None

