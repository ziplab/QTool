
import torch
import torch.nn as nn
import sys
import logging
import numpy as np

if sys.version_info[0] == 3:
    from . import dorefa as dorefa

from .quant import conv3x3, conv1x1, conv0x0

def seq_c_b_a_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = bn(out)
    out = relu(out)
    if skip_enable:
        out += skip
    return out

def seq_c_b_s_a(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = bn(out)
    if skip_enable:
        out += skip
    out = relu(out)
    return out

def seq_c_a_b_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = conv(x)
    out = relu(out)
    out = bn(out)
    if skip_enable:
        out += skip
    return out

def seq_b_c_a_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = bn(x)
    out = conv(out)
    out = relu(out)
    if skip_enable:
        out += skip
    return out

def seq_b_a_c_s(x, conv, relu, bn, skip=None, skip_enable=False):
    out = bn(x)
    out = relu(out)
    out = conv(out)
    if skip_enable:
        out += skip
    return out

def actv(args=None, negative_slope=0.01, clip_at=None):
    keyword = None
    if args is not None:
        keyword = getattr(args, "keyword", None)

    if keyword is None:
        return nn.ReLU(inplace=True)

    if 'PReLU' in keyword:
        return nn.PReLU()

    if 'NReLU' in keyword: # Not with ReLU
        return nn.Sequential()

    if 'ReLU6' in keyword:
        return nn.ReLU6(inplace=True)

    if 'LReLU' in keyword:
        return nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    return nn.ReLU(inplace=True)

# TResNet: High Performance GPU-Dedicated Architecture (https://arxiv.org/pdf/2003.13630v1.pdf)
class TResNetStem(nn.Module):
    def __init__(self, out_channel, in_channel=3, stride=4, kernel_size=1, args=None):
        super(TResNetStem, self).__init__()
        self.stride = stride
        force_fp = True
        if hasattr(args, 'keyword'):
            force_fp = 'real_skip' in args.keyword
        assert kernel_size in [1, 3], "Error reshape conv kernel"
        if kernel_size == 1:
            self.conv = conv1x1(in_channel*stride*stride, out_channel, args=args, force_fp=force_fp)
        elif kernel_size == 3:
            self.conv = conv3x3(in_channel*stride*stride, out_channel, args=args, force_fp=force_fp)

    def forward(self, x):
        B, C, H, W = x.shape
        # consider to employ the PixelShuffle layer instead
        x = x.reshape(B, C, H // self.stride, self.stride, W // self.stride, self.stride)
        x = x.transpose(4, 3).reshape(B, C, 1, H // self.stride, W // self.stride, self.stride * self.stride)
        x = x.transpose(2, 5).reshape(B, C * self.stride * self.stride, H // self.stride, W // self.stride)
        x = self.conv(x)
        return x

class EltWiseModule(torch.nn.Module):
    def __init__(self, operator='sum', args=None):
        super(EltWiseModule, self).__init__()

        # quantization related attributes
        self.enable = False
        self.args = args
        self.index = -1
        self.tag = 'eltwise'
        self.x_index = []
        self.y_index = []
        if self.args is not None:
            logger_root = args.logger_root + '.' if hasattr(args, 'logger_root') else ''
            self.logger = logging.getLogger(logger_root + __name__)
        else:
            self.logger = logging.getLogger(__name__ + '.Quantization')
        self.verbose = self.logger.info
        self.input_index = ""

    def convert_eltwise_to_quantization_version(self, args=None, index=-1):
        if args is not None and import_quantization:
            self.args = args
            self.update_eltwise_quantization_parameter(index=index)

    def update_eltwise_quantization_parameter(self, **parameters):
        index = self.index
        if 'index' in parameters:
            index =  parameters['index']
        if index != self.index:
            self.index = index
            self.verbose('update %s_index %r' % (self.tag, self.index))

        if 'by_index' in parameters:
            by_index = parameters['by_index']
            if isinstance(by_index, list) or (isinstance(by_index, str) and by_index != "all"):
                try:
                    if not isinstance(by_index, list):
                        by_index = by_index.split()
                    by_index = [int(i) for i in by_index]
                except (ValueError, SyntaxError) as e:
                    self.verbose('unexpect string in by_index: {}'.format(by_index))

            if by_index == 'all' or self.index in by_index:
                if ('by_tag' in parameters and self.tag in parameters['by_tag']) or ('by_tag' not in parameters):
                        for k, v in list(parameters.items()):
                            if hasattr(self, "{}".format(k)):
                                if isinstance(v, str):
                                    v = v.replace("'", "").replace('"', '')
                                if isinstance(getattr(self, k), bool):
                                    v = False if v in ['False', 'false', False] else True
                                elif isinstance(getattr(self, k), int):
                                    v = int(v)
                                elif isinstance(getattr(self, k), float):
                                    v = float(v)
                                elif isinstance(getattr(self, k), list) and isinstance(v, str):
                                    v = v.split(',') if ',' in v else v.split(' ')
                                setattr(self, "{}".format(k), v)
                                self.verbose('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))

        if self.enable:
            assert self.args is not None, "args should not be None"
            assert hasattr(self.args, 'global_buffer'), "no global_buffer found in quantization args"

    def coordinate(self, mark_x=0, mark_y=0):
        input_index = self.input_index.split('/')
        if mark_x < len(self.x_index):
            alphaX = self.args.global_buffer[self.x_index[mark_x]]
        elif len(input_index) == 2 and input_index[0] in self.args.global_buffer:
            alphaX = self.args.global_buffer[input_index[0]]
        else:
            self.verbose('cannot find X mark {} for EltWise layer index {}. Disable quantization.'.format(mark_x, self.index))
            return None, None
        if mark_y < len(self.y_index):
            alphaY = self.args.global_buffer[self.y_index[mark_y]]
        elif len(input_index) == 2 and input_index[1] in self.args.global_buffer:
            alphaY = self.args.global_buffer[input_index[1]]
        else:
            self.verbose('cannot find Y mark {} for EltWise layer index {}. Disable quantization '.format(mark_y, self.index))
            return None, None

        assert len(alphaX) == len(alphaY)
        alpha = alphaX / alphaY
        scale = np.ones_like(alpha)
        factor= np.ones_like(alpha)
        for i, x in enumerate(alpha):
            if abs(x) > 1.0:
                factor[i] = alphaY[i]
                scale[i] = x
            else:
                factor[i] = alphaX[i]
                scale[i] = 1. / x
        self.verbose("add add-{} to global_buffer".format(self.index))
        self.args.global_buffer['add-{}'.format(self.index)] = factor

        error = np.ones_like(alpha)
        shift = np.zeros_like(alpha)
        for i in range(16):
            for idx, frac in enumerate(scale):
                tmp = frac * pow(2.0, i)
                cur = abs(round(tmp) - tmp)
                if cur < error[idx]:
                    shift[idx] = i
                    error[idx] = cur

        scaled = [round(s * pow(2., d)) / pow(2., d) for d, s in zip(shift, scale)]
        scale_x = [ scaled[i] / x if abs(x) > 1.0 else 1.0 for i, x in enumerate(alpha)]
        scale_y = [ 1.0 if abs(x) > 1.0 else scaled[i] * x for i, x in enumerate(alpha)]
        return np.array(scale_x), np.array(scale_y)

    def coordinate_addition(self, x, y, mark_x=0, mark_y=0):
        scale_x, scale_y = self.coordinate(mark_x, mark_y)
        if scale_x is None or scale_y is None:
            self.enable = False
            return x, y
        scale_x = torch.from_numpy(scale_x).to(device=x.device, dtype=x.dtype)
        scale_y = torch.from_numpy(scale_y).to(device=x.device, dtype=x.dtype)
        scale_x = scale_x.reshape(1, -1, 1, 1)
        scale_y = scale_y.reshape(1, -1, 1, 1)
        x = x * scale_x
        y = y * scale_y
        return x, y

    def forward(self, x, y, mark_x=0, mark_y=0):
        if self.enable:
            x, y = self.coordinate_addition(x, y, mark_x, mark_y)
        output = x + y
        return output

    def __repr__(self):
        base = 'EltWiseModule()'
        if self.enable:
            base = base + "-index({})-input_index({})".format(self.index, self.input_index)
        return base

def add(args):
    return EltWiseModule(args=args)
    
class Shuffle(nn.Module):
    def __init__(self, groups, args=None):
        super(Shuffle, self).__init__()
        self.groups = groups
        self.index = -1
        self.verbose = print
        self.enable = False
        self.input_index = ""
        self.tag = 'fm'
        self.args = args
        if self.args is not None:
            logger_root = args.logger_root + '.' if hasattr(args, 'logger_root') else ''
            self.logger = logging.getLogger(logger_root + __name__)
        else:
            self.logger = logging.getLogger(__name__ + '.Quantization')
        self.verbose = self.logger.info

    def update_shuffle_quantization_parameter(self, **parameters):
        index = self.index
        if 'index' in parameters:
            index =  parameters['index']
        if index != self.index:
            self.index = index
            self.verbose('update %s_index %r' % (self.tag, self.index))

        if 'by_index' in parameters:
            by_index = parameters['by_index']
            if isinstance(by_index, list) or (isinstance(by_index, str) and by_index != "all"):
                try:
                    if not isinstance(by_index, list):
                        by_index = by_index.split()
                    by_index = [int(i) for i in by_index]
                except (ValueError, SyntaxError) as e:
                    self.verbose('unexpect string in by_index: {}'.format(by_index))

            if by_index == 'all' or self.index in by_index:
                if ('by_tag' in parameters and self.tag in parameters['by_tag']) or ('by_tag' not in parameters):
                        for k, v in list(parameters.items()):
                            if hasattr(self, "{}".format(k)):
                                if isinstance(v, str):
                                    v = v.replace("'", "").replace('"', '')
                                if isinstance(getattr(self, k), bool):
                                    v = False if v in ['False', 'false', False] else True
                                elif isinstance(getattr(self, k), int):
                                    v = int(v)
                                if not isinstance(getattr(self, k), torch.Tensor):
                                    setattr(self, "{}".format(k), v)
                                    self.verbose('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))

    def __repr__(self):
        base = super(Shuffle, self).__repr__()
        if self.enable:
            base = base + "-index({})-input({})".format(self.index, self.input_index)
        return base

    def forward(self, x):
        N, C, H, W = x.shape
        g = self.groups

        if self.enable:
            input_index = self.input_index.split('/')
            scale = []
            for i in input_index:
                if i in self.args.global_buffer:
                    scale.append(self.args.global_buffer[i])
                else:
                    self.verbose("warning {} not found in global_buffer".format(i))
                    
            scaled = np.array(scale)
            scaled = scaled.reshape(g, -1).transpose(1, 0).reshape(-1)
            assert len(scaled) == C and ((C % 2) == 0)

            self.args.global_buffer['shuffle-{}'.format(self.index)] = scaled
            self.verbose("add shuffle-{} to global_buffer".format(self.index))

        # (N C H W) -> (N g C/g H, W) -> (N C/g g H, W) -> (N C H, W)
        return x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

def shuffle(groups, args=None):
    return Shuffle(groups, args=args)

class BatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, args=None):
        super(BatchNorm2d, self).__init__(num_features, eps=1e-5)

        # quantization related attributes
        self.enable = False
        self.args = args
        if self.args is not None:
            logger_root = args.logger_root + '.' if hasattr(args, 'logger_root') else ''
            self.logger = logging.getLogger(logger_root + __name__)
        else:
            self.logger = logging.getLogger(__name__ + '.Quantization')
        self.verbose = self.logger.info
        self.index = -1
        self.input_scale = 1.0
        self.tag = 'norm'
        self.input_index = ""
        def identity(x):
            return x
        self.quant_functions = { "RoundSTE": dorefa.RoundSTE.apply, "identity": identity }
        self.choice = 'identity'

    def update_norm_quantization_parameter(self, **parameters):
        index = self.index
        if 'index' in parameters:
            index =  parameters['index']
        if index != self.index:
            self.index = index
            self.logger.info('update %s_index %r' % (self.tag, self.index))

        if 'by_index' in parameters:
            by_index = parameters['by_index']
            if isinstance(by_index, list) or (isinstance(by_index, str) and by_index != "all"):
                try:
                    if not isinstance(by_index, list):
                        by_index = by_index.split()
                    by_index = [int(i) for i in by_index]
                except (ValueError, SyntaxError) as e:
                    self.verbose('unexpect string in by_index: {}'.format(by_index))

            if by_index == 'all' or self.index in by_index:
                if ('by_tag' in parameters and self.tag in parameters['by_tag']) or ('by_tag' not in parameters):
                        for k, v in list(parameters.items()):
                            if hasattr(self, "{}".format(k)):
                                if isinstance(v, str):
                                    v = v.replace("'", "").replace('"', '')
                                if isinstance(getattr(self, k), bool):
                                    v = False if v in ['False', 'false', False] else True
                                elif isinstance(getattr(self, k), int):
                                    v = int(v)
                                elif isinstance(getattr(self, k), float):
                                    v = float(v)
                                elif isinstance(getattr(self, k), str):
                                    if k == 'input_index':
                                        if 'same' in v:
                                            v = v.replace('same', str(self.index))
                                        elif "last" in v:
                                            v = v.replace('last', str(self.index-1))
                                if not isinstance(getattr(self, k), torch.Tensor):
                                    setattr(self, "{}".format(k), v)
                                    self.verbose('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))
                                    if k == 'input_index':
                                        for tag in ['fm', 'wt']:
                                            exist = 'Yes' if self.input_index + "-{}".format(tag) in self.args.global_buffer else 'No'
                                            self.verbose("input_index of tag({}) exist in global buffer ? {}".format(tag, exist))
        if self.enable:
            assert self.args is not None, "args should not be None"
            assert hasattr(self.args, 'global_buffer'), "no global_buffer found in quantization args"
            assert self.choice in self.quant_functions, "unknown choice of quant_function {}".format(self.choice)

    def forward(self, x):
        if x.numel() > 0:
            if self.enable and 'skip' not in self.input_index:
                scale = self.weight * (self.running_var + self.eps).rsqrt()
                bias = self.bias / scale - self.running_mean
                input_scale = self.input_scale
                if self.input_index + "-fm" in self.args.global_buffer:
                    input_scale = input_scale * self.args.global_buffer[self.input_index + "-fm"].abs().item()
                if self.input_index + "-wt" in self.args.global_buffer:
                    input_scale = input_scale * self.args.global_buffer[self.input_index + "-wt"].abs().item()
                if self.input_index + "-norm" not in self.args.global_buffer:
                    self.verbose("add clip_val-{}-norm to global_buffer".format(self.index))
                self.args.global_buffer["clip_val-{}-norm".format(self.index)] = input_scale * scale.cpu().detach().numpy()
                bias = self.quant_functions[self.choice](bias / input_scale) * input_scale
                scale = scale.reshape(1, -1, 1, 1)
                bias = bias.reshape(1, -1, 1, 1)
                return (x + bias) * scale
            else:
                return super(BatchNorm2d, self).forward(x)
        else:
            raise RuntimeError("should not reach here")

    def __repr__(self):
        base = super(BatchNorm2d, self).__repr__()
        if self.enable:
            base = base + "-choice({})-index({})-input({})".format(self.choice, self.index, self.input_index)
        return base

class FrozenBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class StaticBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics are fixed, but the affine parameters are not.
    """
    def __init__(self, num_features, eps=1e-5, args=None):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features) - eps)

    def forward(self, x):
        scale = self.weight * (self.running_var + self.eps).rsqrt()
        bias = self.bias - self.running_mean * scale
        scale = scale.reshape(1, -1, 1, 1)
        bias = bias.reshape(1, -1, 1, 1)
        return x * scale + bias

class ReverseBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__(num_features, eps=eps, affine=False)
        assert affine, "Affine should be True for ReverseBatchNorm2d"
        self.weight = nn.Parameter(torch.ones(num_features), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(num_features), requires_grad=True)

    def forward(self, x):
        scale = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        x = x * scale + bias
        x = super(ReverseBatchNorm2d, self).forward(x)
        return x

def norm(channel, eps=1e-5, args=None, keyword=None, feature_stride=None, affine=False):
    if args is not None and keyword is None:
        keyword = getattr(args, "keyword", None)

    if keyword is None:
        return nn.BatchNorm2d(channel)

    if "group-norm" in keyword:
        group = getattr(args, "fm_quant_group", 32)
        return nn.GroupNorm(group, channel)

    if "static-bn" in keyword:
        return StaticBatchNorm2d(channel, args=args)

    if "freeze-bn" in keyword:
        return FrozenBatchNorm2d(channel)

    if "reverse-bn" in keyword:
        return ReverseBatchNorm2d(channel)

    if "sync-bn" in keyword:
        return nn.SyncBatchNorm(channel)

    if "quant-bn" in keyword:
        return BatchNorm2d(channel, args=args)

    if "instance-norm" in keyword:
        return nn.InstanceNorm2d(channel, affine=affine)

    return nn.BatchNorm2d(channel)

class Split(nn.Module):
    def __init__(self, dim=1, first_half=True, args=None):
        super(Split, self).__init__()
        assert(dim == 1)
        self.first_half = first_half
        self.index = -1
        self.verbose = print
        self.enable = False
        self.input_index = ""
        self.tag = 'fm'
        self.args = args
        if self.args is not None:
            logger_root = args.logger_root + '.' if hasattr(args, 'logger_root') else ''
            self.logger = logging.getLogger(logger_root + __name__)
        else:
            self.logger = logging.getLogger(__name__ + '.Quantization')
        self.verbose = self.logger.info

    def update_split_quantization_parameter(self, **parameters):
        index = self.index
        if 'index' in parameters:
            index =  parameters['index']
        if index != self.index:
            self.index = index
            self.verbose('update %s_index %r' % (self.tag, self.index))

        if 'by_index' in parameters:
            by_index = parameters['by_index']
            if isinstance(by_index, list) or (isinstance(by_index, str) and by_index != "all"):
                try:
                    if not isinstance(by_index, list):
                        by_index = by_index.split()
                    by_index = [int(i) for i in by_index]
                except (ValueError, SyntaxError) as e:
                    self.verbose('unexpect string in by_index: {}'.format(by_index))

            if by_index == 'all' or self.index in by_index:
                if ('by_tag' in parameters and self.tag in parameters['by_tag']) or ('by_tag' not in parameters):
                        for k, v in list(parameters.items()):
                            if hasattr(self, "{}".format(k)):
                                if isinstance(v, str):
                                    v = v.replace("'", "").replace('"', '')
                                if isinstance(getattr(self, k), bool):
                                    v = False if v in ['False', 'false', False] else True
                                elif isinstance(getattr(self, k), int):
                                    v = int(v)
                                if not isinstance(getattr(self, k), torch.Tensor):
                                    setattr(self, "{}".format(k), v)
                                    self.verbose('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))

    def __repr__(self):
        base = super(Split, self).__repr__()
        if self.enable:
            base = base + "-first_half({})-index({})-input({})".format(self.first_half, self.index, self.input_index)
        return base

    def forward(self, x):
        N, C, H, W = x.shape
        if self.enable:
            input_index = self.input_index.split('/')
            scale = []
            for i in input_index:
                if i in self.args.global_buffer:
                    scale.append(self.args.global_buffer[i])
                else:
                    self.verbose("warning {} not found in global_buffer".format(i))
                    
            scaled = np.array(scale)
            scaled = scaled.reshape(-1)
            #if self.index in [6, 7]:
            #    import pdb
            #    pdb.set_trace()
            if self.first_half:
                self.args.global_buffer['split-{}'.format(self.index)] = scaled[:C//2]
                self.verbose("add split-{} to global_buffer".format(self.index))
            else:
                self.args.global_buffer['split-{}'.format(self.index)] = scaled[C//2:]
                self.verbose("add split-{} to global_buffer".format(self.index))

        splits = torch.chunk(x, 2, dim=1)
        return splits[0] if self.first_half else splits[1]

def split(dim, first_half, args=None):
    return Split(dim=dim, first_half=first_half, args=args)


class Concat(nn.Module):
    def __init__(self, args=None):
        super(Concat, self).__init__()
        self.index = -1
        self.verbose = print
        self.enable = False
        self.input_index = ""
        self.tag = 'fm'
        self.args = args
        if self.args is not None:
            logger_root = args.logger_root + '.' if hasattr(args, 'logger_root') else ''
            self.logger = logging.getLogger(logger_root + __name__)
        else:
            self.logger = logging.getLogger(__name__ + '.Quantization')
        self.verbose = self.logger.info

    def update_concat_quantization_parameter(self, **parameters):
        index = self.index
        if 'index' in parameters:
            index =  parameters['index']
        if index != self.index:
            self.index = index
            self.verbose('update %s_index %r' % (self.tag, self.index))

        if 'by_index' in parameters:
            by_index = parameters['by_index']
            if isinstance(by_index, list) or (isinstance(by_index, str) and by_index != "all"):
                try:
                    if not isinstance(by_index, list):
                        by_index = by_index.split()
                    by_index = [int(i) for i in by_index]
                except (ValueError, SyntaxError) as e:
                    self.verbose('unexpect string in by_index: {}'.format(by_index))

            if by_index == 'all' or self.index in by_index:
                if ('by_tag' in parameters and self.tag in parameters['by_tag']) or ('by_tag' not in parameters):
                        for k, v in list(parameters.items()):
                            if hasattr(self, "{}".format(k)):
                                if isinstance(v, str):
                                    v = v.replace("'", "").replace('"', '')
                                if isinstance(getattr(self, k), bool):
                                    v = False if v in ['False', 'false', False] else True
                                elif isinstance(getattr(self, k), int):
                                    v = int(v)
                                if not isinstance(getattr(self, k), torch.Tensor):
                                    setattr(self, "{}".format(k), v)
                                    self.verbose('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))

    def __repr__(self):
        base = super(Concat, self).__repr__()
        if self.enable:
            base = base + "-index({})-input({})".format(self.index, self.input_index)
        return base

    def forward(self, x, y):
        N, C, H, W = x.shape
        if self.enable:
            input_index = self.input_index.split('/')
            scale = []
            for i in input_index:
                if i in self.args.global_buffer:
                    scale = scale + self.args.global_buffer[i].tolist()
                else:
                    self.verbose("warning {} not found in global_buffer".format(i))

            scaled = np.array(scale)
            scaled = scaled.reshape(-1)
            #if self.index in [4]:
            #    import pdb
            #    pdb.set_trace()
            self.args.global_buffer['concat-{}'.format(self.index)] = scaled
            self.verbose("add concat-{} to global_buffer".format(self.index))

        return torch.cat((x, y), dim=1)

def concat(args=None):
    return Concat(args=args)


