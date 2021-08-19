import os, sys, glob, argparse
import logging
import types
from collections import OrderedDict

import torch
import torch.nn.functional as F

import utils
import models
import main as entry

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def export_onnx(args):
    model_name = args.model
    if model_name in models.model_zoo:
        model, args = models.get_model(args)
    else:
        print("model(%s) not support, available models: %r" % (model_name, models.model_zoo))
        return

    if utils.check_file(args.old):
        print("load pretrained from %s" % args.old)
        if torch.cuda.is_available():
            checkpoint = torch.load(args.old)
        else:  # force cpu mode
            checkpoint = torch.load(args.old, map_location='cpu')
        print("load pretrained ==> last epoch: %d" % checkpoint.get('epoch', 0))
        print("load pretrained ==> last best_acc: %f" % checkpoint.get('best_acc', 0))
        print("load pretrained ==> last learning_rate: %f" % checkpoint.get('learning_rate', 0))
        try:
            utils.load_state_dict(model, checkpoint.get('state_dict', checkpoint))
        except RuntimeError:
            print("Loading pretrained model failed")
    else:
        print("no pretrained file exists({}), init model with default initlizer".
            format(args.old))

    onnx_model = torch.nn.Sequential(OrderedDict([
        ('network', model),
        #('softmax', torch.nn.Softmax()),
    ]))

    onnx_path = "onnx/" + model_name
    if not os.path.exists(onnx_path):
        os.makedirs(onnx_path)
    onnx_save = onnx_path + "/" + model_name + '.onnx'

    input_names = ["input"]
    dummy_input = torch.zeros((1, 3, args.input_size, args.input_size))
    output_names = ['prob']
    torch.onnx.export(
            onnx_model,
            dummy_input,
            onnx_save,
            verbose=True,
            input_names=input_names,
            output_names=output_names,
            opset_version=7,
            keep_initializers_as_inputs=True
            )

def inference(args):
    from models.quant import custom_conv
    def init(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
            args=None, force_fp=False, feature_stride=1):
        super(custom_conv, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.args = args
        self.force_fp = True

    custom_conv.__init__ = init

    model_name = args.model
    if model_name in models.model_zoo:
        model, args = models.get_model(args)
    else:
        print("model(%s) not support, available models: %r" % (model_name, models.model_zoo))
        return

    def forward(self, x):
        print(x.shape, self.weight.shape, self.kernel_size, self.stride, self.padding, self.dilation, self.groups)
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return output

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.forward = types.MethodType(forward, m)

    input = torch.rand(1, 3, args.input_size, args.input_size)
    model.forward(input)

def get_parameter():
    parser = entry.get_parser()
    parser.add_argument('--old', type=str, default='')
    parser.add_argument('--new', type=str, default='')
    parser.add_argument('--mapping_from', '--mf', type=str, default='')
    parser.add_argument('--mapping_to', '--mt', type=str, default='')
    parser.add_argument('--verbose_list', default='ratio,sep', type=str)
    args = parser.parse_args()
    if isinstance(args.verbose_list, str):
        args.verbose_list = [x.strip() for x in args.verbose_list.split(',')]
    if isinstance(args.keyword, str):
        args.keyword = [x.strip() for x in args.keyword.split(',')]
    return args

def main():
    args = get_parameter()
    args.weights_dir = os.path.join(args.weights_dir, args.model)
    utils.check_folder(args.weights_dir)

    if os.path.exists(args.log_dir):
        utils.setup_logging(os.path.join(args.log_dir, 'tools.txt'), resume=True)

    config = dict()
    for i in args.keyword:
        config[i] = True

    if 'export_onnx' in config.keys():
        export_onnx(args)

    if 'inference' in config.keys():
        inference(args)

    if 'verbose' in config.keys():
        if torch.cuda.is_available():
            checkpoint = torch.load(args.old)
        else:  # force cpu mode
            checkpoint = torch.load(args.old, map_location='cpu')
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        for name, value in checkpoint.items():
            if ('quant_activation' in name or 'quant_weight' in name) and name.split('.')[-1] in args.verbose_list:
                print(name, value.shape, value.requires_grad)
                print(value.data)
            elif "all" in args.verbose_list:
                if 'num_batches_tracked' not in name:
                    if isinstance(value, torch.Tensor):
                        print(name, value.shape, value.requires_grad)
                    elif isinstance(value, int) or isinstance(value, float) or isinstance(value, str):
                        print(name, value, type(value))
                    else:
                        print(name, type(value))

    if 'load' in config.keys() or 'save' in config.keys():
        model_name = args.model
        if model_name in models.model_zoo:
            model, args = models.get_model(args)
        else:
            print("model(%s) not support, available models: %r" % (model_name, models.model_zoo))
            return
        if utils.check_file(args.old):
            raw = 'raw' in config.keys()
            if torch.cuda.is_available():
                checkpoint = torch.load(args.old)
            else:  # force cpu mode
                checkpoint = torch.load(args.old, map_location='cpu')
            try:
                utils.load_state_dict(model, checkpoint.get('state_dict', None) if not raw else checkpoint, verbose=False)
            except RuntimeError:
                print("Loading pretrained model failed")
            print("Loading pretrained model OK")

            if 'save' in config.keys() and args.new != '':
                torch.save(model.state_dict(), args.new)
                print("Save pretrained model into %s" % args.new)
        else:
            print("file not exist %s" % args.old)

    if 'update' in config.keys():
        mapping_from = []
        mapping_to = []
        if os.path.isfile(args.mapping_from):
            with open(args.mapping_from) as f:
                mapping_from = f.readlines()
                f.close()
        if os.path.isfile(args.mapping_to):
            with open(args.mapping_to) as f:
                mapping_to = f.readlines()
                f.close()
        mapping_from = [ i.strip().strip('\n').strip('"').strip("'") for i in mapping_from]
        mapping_from = [ i for i in mapping_from if len(i) > 0 and i[0] != '#'] 
        mapping_to = [ i.strip().strip('\n').strip('"').strip("'") for i in mapping_to]
        mapping_to = [ i for i in mapping_to if len(i) > 0 and i[0] != '#']
        if len(mapping_to) != len(mapping_from) or len(mapping_to) == 0 or len(mapping_from) == 0:
            mapping = None
            logging.info('no valid mapping')
        else:
            mapping = dict()
            for i, k in enumerate(mapping_from):
                if '{' in k and '}' in k and '{' in mapping_to[i] and '}' in mapping_to[i]:
                    item = k.split('{')
                    for v in item[1].strip('}').split(","):
                        v = v.strip()
                        mapping[item[0] + v] = mapping_to[i].split('{')[0] + v
                else:
                    mapping[k] = mapping_to[i] 

        raw = 'raw' in config.keys()
        if not os.path.isfile(args.old):
            args.old = args.pretrained
        utils.import_state_dict(args.old, args.new, mapping, raw, raw_prefix=args.case)

    if 'det-load' in  config.keys():
        from third_party.checkpoint import DetectionCheckpointer
        model_name = args.model
        if model_name in models.model_zoo:
            model, args = models.get_model(args)
        else:
            print("model(%s) not support, available models: %r" % (model_name, models.model_zoo))
            return
        split = os.path.split(args.old)
        checkpointer = DetectionCheckpointer(model, split[0], save_to_disk=True)
        checkpointer.resume_or_load(args.old, resume=True)
        checkpointer.save(split[1])

    if 'swap' in config.keys():
        mapping_from = []
        if os.path.isfile(args.mapping_from):
            with open(args.mapping_from) as f:
                mapping_from = f.readlines()
                f.close()
            mapping_from = [ i.strip().strip('\n').strip('"').strip("'") for i in mapping_from]
            mapping_from = [ i for i in mapping_from if len(i) > 0 and i[0] != '#']
            lists = args.verbose_list
            for i in lists:
                item = i.split('/')
                interval = (int)(item[0])
                index = item[1].split('-')
                index = [(int)(x) for x in index]
                if len(mapping_from) % interval == 0 and len(index) <= interval:
                    mapping_to = mapping_from.copy()
                    for j, k in enumerate(index):
                        k = k % interval
                        mapping_to[j::interval] = mapping_from[k::interval]

            mapping_to= [ i + '\n' for i in mapping_to]
            with open(args.mapping_from + "-swap", 'w') as f:
                f.writelines(mapping_to)
                f.close()

    if 'sort' in config.keys():
        mapping_from = []
        if os.path.isfile(args.mapping_from):
            with open(args.mapping_from) as f:
                mapping_from = f.readlines()
                f.close()
            mapping_from.sort()
            with open(args.mapping_from + "-sort", 'w') as f:
                f.writelines(mapping_from)
                f.close()

    if 'verify-data' in config.keys() or 'verify-image' in config.keys():
        if 'verify-image' in config.keys():
            lists = args.verbose_list
        else:
            with open(os.path.join(args.root, 'train.txt')) as f:
                lists = f.readlines()
                f.close()
        from PIL import Image
        from threading import Thread
        print("going to check %d files" % len(lists))
        def check(lists, start, end, index):
            for i, item in enumerate(lists[start:end]):
                try:
                    items = item.split()
                    if len(items) >= 1:
                        path = items[0].strip().strip('\n')
                    else:
                        print("skip line %s" % i)
                        continue
                    path = os.path.join(args.root, os.path.join("train", path))
                    imgs = Image.open(path)
                    imgs.resize((256,256))
                    if index == 0:
                        print(i, end ="\r", file=sys.stderr)
                except (RuntimeError, IOError):
                    print("\nError when read image %s" % path)
            print("\nFinish checking", index)
        #lists = lists[45000:]
        num = min(len(lists), 20)
        for i in range(num):
            start = len(lists) // num * i
            end = min(start + len(lists) // num, len(lists))
            th = Thread(target=check, args=(lists, start, end, i))
            th.start()

if __name__ == '__main__':
    main()

