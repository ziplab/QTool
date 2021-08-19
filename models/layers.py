
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import logging
import numpy as np
import pdb

if sys.version_info[0] == 3:
    from . import dorefa as dorefa
    from . import discretization as disc
    from . import save_tensor as st

from .quant import conv3x3, conv1x1, conv0x0, quantization

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

class custom_relu(nn.ReLU):
    def __init__(self, inplace=False, args=None):
        super(custom_relu, self).__init__(inplace)
        self.args = args
        self.quant_activation = quantization(args, 'fm')
        self.force_fp = False

    def update_relu_quantization_parameter(self, **parameters):
        if not self.force_fp:
            feedback = dict()
            def merge_dict(feedback, fd):
                if fd is not None:
                    for k in fd:
                        if k in feedback:
                            if isinstance(fd[k], list) and isinstance(feedback[k], list):
                                feedback[k] = feedback[k] + fd[k]
                        else:
                            feedback[k] = fd[k]
            fd = self.quant_activation.update_quantization(**parameters)
            merge_dict(feedback, fd)
            return feedback
        else:
            return None

    def forward(self, inputs):
        output = F.relu(inputs, inplace=self.inplace)
        output = self.quant_activation(output)
        return output

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

    if 'QReLU' in keyword:
        return custom_relu(inplace=True, args=args)

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
        if self.args is not None:
            logger_root = args.logger_root + '.' if hasattr(args, 'logger_root') else ''
            self.logger = logging.getLogger(logger_root + __name__)
        else:
            self.logger = logging.getLogger(__name__ + '.Quantization')
        self.verbose = self.logger.info

        # borrow setting via assigning tag = 'fm'
        tag = 'fm'
        self.method = 'none'
        self.choice = 'none'
        self.__EPS__ = 0.0
        self.num_levels = getattr(args, tag + '_level', None)
        self.bit = getattr(args, tag + '_bit', None)
        self.half_range = getattr(args, tag + '_half_range', None)
        self.quant_group = getattr(args, tag + '_quant_group', None)
        self.boundary = getattr(self.args, tag + '_boundary', None)
        self.clip_val = nn.Parameter(torch.Tensor([self.boundary]))
        self.quant = dorefa.LSQ
        self.clamp = torch.clamp
        self.choice = 'lsq'
        self.quant_output = True
        self.quant_input = True

        if self.num_levels is None or self.num_levels <= 0:
            self.num_levels = int(2 ** self.bit)

        self.iteration = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.stable = getattr(args, tag + '_stable', 0)
        if self.stable < 0:
            self.stable = getattr(args, 'stable', 0)

        self.input_index = ""
        self.shared_clip = ""
        self.x_index = []
        self.y_index = []
        self.name = ""

        self.bit_limit = 0

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

    def coordinate(self, mark_x=0, mark_y=0, input1=None, input2=None):
        if hasattr(self, 'shared_clip') and self.shared_clip != "":
            clip_val = self.args.global_buffer[self.shared_clip].abs()
        else:
            clip_val = self.clip_val.abs()

        step = clip_val.item() / (self.num_levels - 1)
        if self.quant_input:
            if 'add-{}_alphaX'.format(self.index) not in self.args.global_buffer or \
               'add-{}_alphaY'.format(self.index) not in self.args.global_buffer:
                input_index = self.input_index.split('/')
                if mark_x < len(self.x_index):
                    input_index_x = self.x_index[mark_x]
                elif len(input_index) == 2 and input_index[0] in self.args.global_buffer:
                    input_index_x = input_index[0]
                else:
                    self.verbose('cannot find X mark {} for EltWise layer index {}. Disable quantization.'.format(mark_x, self.index))
                    raise RuntimeError("unexpected index_index for eltwise {}".format(self.index))
                if mark_y < len(self.y_index):
                    input_index_y = self.y_index[mark_y]
                elif len(input_index) == 2 and input_index[1] in self.args.global_buffer:
                    input_index_y = input_index[1]
                else:
                    self.verbose('cannot find Y mark {} for EltWise layer index {}. Disable quantization '.format(mark_y, self.index))
                    raise RuntimeError("unexpected index_index for eltwise {}".format(self.index))

                alphaX = self.args.global_buffer[input_index_x]
                alphaY = self.args.global_buffer[input_index_y]
                if isinstance(alphaX, torch.Tensor):
                    alphaX = alphaX.cpu().numpy()
                if isinstance(alphaY, torch.Tensor):
                    alphaY = alphaY.cpu().numpy()
                self.args.global_buffer['add-{}_alphaX'.format(self.index)] = alphaX
                self.args.global_buffer['add-{}_alphaY'.format(self.index)] = alphaY
                self.args.global_buffer['add-{}_input_index_x'.format(self.index)] = input_index_x
                self.args.global_buffer['add-{}_input_index_y'.format(self.index)] = input_index_y
            else:
                alphaX = self.args.global_buffer['add-{}_alphaX'.format(self.index)]
                alphaY = self.args.global_buffer['add-{}_alphaY'.format(self.index)]
                input_index_x = self.args.global_buffer['add-{}_input_index_x'.format(self.index)]
                input_index_y = self.args.global_buffer['add-{}_input_index_y'.format(self.index)]

            assert len(alphaX) == len(alphaY)

            self.bit_limit = 8
            eta_x, shifts_x, CF_x, offset_x = disc.discretize(input1, alphaX, step, self.num_levels, self.bit_limit, \
                        global_buffer=self.args.global_buffer, index=self.index, input_index=input_index_x, \
                        closed_form=False)
            eta_y, shifts_y, CF_y, offset_y = disc.discretize(input2, alphaY, step, self.num_levels, self.bit_limit, \
                        global_buffer=self.args.global_buffer, index=self.index, input_index=input_index_y, \
                        closed_form=False)
            eta_z = eta_x + eta_y
            output = eta_z.mul(step)

            if 'cmr-add-{}'.format(self.index) not in self.args.global_buffer:
                cmr = max(max(CF_x), max(CF_y))
                self.args.global_buffer['cmr-add-{}'.format(self.index)] = cmr
                self.verbose('cmr-add-{} is: {:4d}'.format(self.index, cmr))
            self.args.global_buffer['add-{}'.format(self.index)] = [step]
        else:
            output = input1 + input2

        if self.quant_output:
            eta_z = output.div(step)
            eta_z = torch.round(eta_z)
            eta_z = torch.round(eta_z*0.75)
            eta_z = torch.clamp(eta_z, min=0, max=self.num_levels - 1)
            output = eta_z.mul(step/0.75)
            self.args.global_buffer['add-{}'.format(self.index)] = [step/0.75]

            if 'print' in self.args.keyword and self.name != "":
                sv = True
                sv = st.tensor_to_txt(eta_z, 4, signed=False, filename='cmodel/output-fm-of-layer-{}.hex'.format(self.name))
                if sv:
                   self.verbose("saving file {}, shape: {}".format('cmodel/output-fm-of-layer-{}.hex'.format(self.name), eta_z.shape))
                else:
                   self.verbose("saving file {} failed".format('cmodel/output-fm-of-layer-{}.hex'.format(self.name)))
                
        #pdb.set_trace()
        return output

    def forward(self, x, y, mark_x=0, mark_y=0):
        self.iteration.data = self.iteration.data + 1
        if self.enable:
            if self.iteration.data <= self.stable and self.training:
                if not self.training:
                    self.verbose("call init_based_on_warmup during testing might indicate error in eltwise-add")
                return x + y

            if 'eval' in self.args.keyword and not self.training and 'skip' not in self.input_index:
                return self.coordinate(mark_x, mark_y, x, y)

            if hasattr(self, 'shared_clip') and self.shared_clip != "":
                clip_val = self.args.global_buffer[self.shared_clip].abs()
            else:
                clip_val = self.clip_val.abs()

            level = self.num_levels - 1
            if self.quant_input:
                x = x.div(clip_val)
                x = self.clamp(x, min=0, max=1)
                x = self.quant.apply(x, level)
                y = y.div(clip_val)
                y = self.clamp(y, min=0, max=1)
                y = self.quant.apply(y, level)
                z = x * clip_val + y * clip_val
            else:
                z = x + y

            if self.quant_output:
                # update 21.07.28 / update 03.08.2021 revise round(1/2.) to round(1/2.) + round(1/4.)
                # update 06.08.2021 round(0.75)
                z = z.div(clip_val/level/0.75)
                #z1 = dorefa.RoundSTE.apply(z * 0.5)
                #z2 = dorefa.RoundSTE.apply(z * 0.25)
                #z = z1 + z2
                z = dorefa.RoundSTE.apply(z)
                z = self.clamp(z, min=0, max=level)
                #z = z.mul(clip_val/level/0.75)
                z = z.mul(clip_val/level/0.75)
            return z
        else:
            return x + y

    def __repr__(self):
        base = 'EltWiseModule()'
        if self.enable:
            base = base + "-index({})-input_index({})-level({})-quant_output({})-quant_input({})-shared_clip-({})".format(
                self.index, self.input_index, self.num_levels, self.quant_output, self.quant_input, self.shared_clip)
            base = base + "-stable({})-iteration({})-name({})".format(self.stable, self.iteration.item(), self.name)
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
        self.name = ""
        self.tag = 'shuffle'
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
            base = base + "-index({})-input_index({})-name({})".format(self.index, self.input_index, self.name)
        return base

    def forward(self, x):
        N, C, H, W = x.shape
        g = self.groups

        # (N C H W) -> (N g C/g H, W) -> (N C/g g H, W) -> (N C H, W)
        y = x.view(N, g, int(C / g), H, W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

        if self.enable and 'eval' in self.args.keyword and self.input_index != "":
            scaled = None
            if 'shuffle-{}'.format(self.index) not in self.args.global_buffer:
                assert '/' not in self.input_index
                if self.input_index in self.args.global_buffer:
                    tmp = self.args.global_buffer[self.input_index]
                    if isinstance(tmp, torch.Tensor):
                        tmp = tmp.cpu().numpy()
                    if len(tmp) == 1:
                        scale = [tmp[0] for i in range(C)]
                    elif len(tmp) == C:
                        scale = tmp.tolist()
                    else:
                        print("unexpected length of scale", len(tmp))
                        pdb.set_trace()
                else:
                    self.verbose("warning {} not found in global_buffer, disable shuffle layer-{}".format(
                        i, self.index))
                    self.enable = False

                if self.enable:
                    #pdb.set_trace()
                    scaled = np.array(scale)
                    scaled = scaled.reshape(g, -1).transpose(1, 0).reshape(-1)
                    assert len(scaled) == C and ((C % 2) == 0)

                    self.args.global_buffer['shuffle-{}'.format(self.index)] = scaled
                    #self.verbose("add shuffle-{} to global_buffer".format(self.index))
            else:
                scaled = self.args.global_buffer['shuffle-{}'.format(self.index)]

            if scaled is not None:
                if 'print' in self.args.keyword and self.name != "":
                    #pdb.set_trace()
                    scaled = scaled / 15.0
                    scaled = torch.from_numpy(scaled).to(device=y.device, dtype=y.dtype)
                    scaled = scaled.reshape(1, -1, 1, 1)
                    eta_z = y.div(scaled).round()
                    sv = True
                    sv = st.tensor_to_txt(eta_z, 4, signed=False, filename='cmodel/output-fm-of-layer-{}.hex'.format(self.name))
                    if sv:
                       self.verbose("saving file {}, shape: {}".format('cmodel/output-fm-of-layer-{}.hex'.format(self.name), eta_z.shape))
                    else:
                       self.verbose("saving file {} failed".format('cmodel/output-fm-of-layer-{}.hex'.format(self.name)))
                
        return y

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
        self.bit = 20
        self.level_num = int(pow(2., self.bit))
        def identity(x):
            return x
        self.quant_functions = { "RoundSTE": dorefa.RoundSTE.apply, "identity": identity }
        self.choice = 'identity'
        self.items = ['tag', 'index', 'input_index', 'choice', 'bit']

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
            if self.enable and 'skip' not in self.input_index and 'eval' in self.args.keyword:
                if self.index not in [0] and False:
                    return super(BatchNorm2d, self).forward(x)
                elif False:
                    alpha = self.weight * (self.running_var + self.eps).rsqrt()
                    beta = self.bias / alpha  - self.running_mean
                    alpha = alpha.reshape(1, -1, 1, 1)
                    beta = beta.reshape(1, -1, 1, 1)
                    return (x + beta) * alpha
                    
                if "norm-{}".format(self.index) in self.args.global_buffer:
                    input_scale = self.args.global_buffer["norm-{}-input_scale".format(self.index)]
                    eta_bias = self.args.global_buffer["norm-{}-eta_bias".format(self.index)]
                    alpha = self.args.global_buffer["norm-{}-alpha".format(self.index)]
                    mask = self.args.global_buffer["norm-{}-mask".format(self.index)]
                else:
                    input_scale = 1./255 if self.index == 0 else 1.
                    input_index_list = self.input_index.split('/')
                    assert len(input_index_list) in [1, 2]

                    input_index = input_index_list[0] + '-wt'
                    if input_index in self.args.global_buffer:
                        if isinstance(self.args.global_buffer[input_index], torch.Tensor):
                            tmp_scale = self.args.global_buffer[input_index].abs().item()
                        else:
                            tmp_scale = self.args.global_buffer[input_index]
                        tmp_scale = tmp_scale[0] if isinstance(tmp_scale, list) else tmp_scale
                        input_scale = input_scale * tmp_scale / 8.
                    else:
                        self.verbose("input_scale {} not found for norm-{}".format(input_index, self.index))

                    input_index = input_index_list[1] if len(input_index_list) == 2 else input_index_list[0] + '-fm'
                    if input_index in self.args.global_buffer:
                        if isinstance(self.args.global_buffer[input_index], torch.Tensor):
                            tmp_scale = self.args.global_buffer[input_index].abs().item()
                        else:
                            tmp_scale = self.args.global_buffer[input_index]
                        # all tmp_scale should be the same if isinstance of np.ndarray
                        tmp_scale = tmp_scale[0] if isinstance(tmp_scale, (list, np.ndarray)) else tmp_scale
                        input_scale = input_scale * tmp_scale
                        if 'add' not in input_index:
                            input_scale = input_scale / 15.
                    else:
                        self.verbose("input_scale {} not found for norm-{}".format(input_index, self.index))

                    alpha = self.weight * (self.running_var + self.eps).rsqrt()
                    beta = self.bias / alpha  - self.running_mean
                    scale = alpha.clone()
                    mask = scale.clone()
                    #pdb.set_trace()
                    for i, s in enumerate(scale):
                        s = alpha[i].abs().item()
                        if s < pow(10., -19):
                            scale[i] = input_scale
                            alpha[i] = 1.
                            beta[i] = self.bias[i] / input_scale
                            mask[i] = 0.
                            self.verbose("clear channel {} in layer norm-{}, as scale ({:.2e}) smaller than 10e-19".format(i, self.index, s))
                            #x[0][i].fill_(0.)
                        else:
                            scale[i] = s * input_scale
                            beta[i] = beta[i] / input_scale
                            mask[i] = 1.

                    #if self.index == 171:
                    #    channel = 42
                    #    shrink = 3.
                    #    beta[channel].mul_(pow(2., -shrink))
                    #    scale[channel].mul_(pow(2., shrink))
                    #    alpha[channel].mul_(pow(2., shrink))
                    #    # remember to decrease the weight for self.input_index layer by pow(2., shrink)

                    eta_bias = beta
                    eta_bias = torch.round(eta_bias)
                    for i, s in enumerate(eta_bias):
                        s = eta_bias[i].abs().item()
                        if s > pow(2., 14): # and False:
                            shrink = np.ceil(np.log2(s / pow(2., 14)))
                            scale[i] = scale[i] * pow(2., shrink)
                            alpha[i] = alpha[i] * pow(2., shrink)
                            eta_bias[i] = torch.round(eta_bias[i] / pow(2., shrink))
                            if shrink >= 1: # and False:
                                #if 'print' in self.args.keyword:
                                self.verbose("clear channel {} in layer norm-{}, as bias requires to shrink {} bit".format(
                                    i, self.index, shrink))
                                mask[i] = 0.
                            else:
                                mask[i] = mask[i] / pow(2., shrink)
                                self.verbose("bias might exceed range, eta_bias[{}]={}, shrink {} bit, after shrink: {}. id: {}- {}".format(
                                    i, s, shrink, eta_bias[i].item(), self.index, self.input_index))
                                pdb.set_trace()

                    self.args.global_buffer["norm-{}-input_scale".format(self.index)] = input_scale
                    self.args.global_buffer["norm-{}-eta_bias".format(self.index)] = eta_bias
                    self.args.global_buffer["norm-{}-alpha".format(self.index)] = alpha
                    self.args.global_buffer["norm-{}-mask".format(self.index)] = mask
                    self.args.global_buffer["norm-{}".format(self.index)] = scale

                #if self.index in [88]:
                #    import pdb
                #    pdb.set_trace()
                eta_conv = x / input_scale
                eta_conv = torch.round(eta_conv)
                eta_conv = eta_conv.mul(mask.reshape(1, -1, 1, 1))
                eta_conv = eta_conv.to(torch.int).to(torch.float32)
                #eta_conv = torch.clamp(eta_conv, min=-self.level_num//2, max=self.level_num-1-self.level_num//2)
                #pdb.set_trace()

                input_index_list = self.input_index.split('/')
                input_index = input_index_list[0]
                if 'print' in self.args.keyword and 'print-{}'.format(input_index) in self.args.global_buffer:
                    self.verbose("append bias to layer {} in norm-{}".format(input_index, self.index))
                    #self.args.global_buffer.pop('print-{}'.format(self.input_index))
                    #print("bias") # {}".format(eta_bias.shape[0]))
                    save_bias = eta_bias.cpu().numpy().astype(np.int)
                    #print(' '.join(map(str, save_bias)))
                    self.args.global_buffer['print-norm-{}-bias'.format(self.index)] = save_bias
                    self.args.global_buffer['print-norm-{}-mask'.format(self.index)] = mask.cpu().numpy().astype(np.int)
                    self.args.global_buffer['print-norm-{}'.format(self.index)] = self.args.global_buffer['print-{}'.format(input_index)]
                    self.args.global_buffer['print-norm-{}-weight'.format(self.index)] = self.args.global_buffer['print-{}-weight'.format(input_index)]

                    name = self.args.global_buffer['print-{}'.format(input_index)]
                    sv = True
                    sv = st.tensor_to_txt(eta_conv, 20, signed=True, filename='cmodel/acc_just_after_layer_{}_and_before_add_bias.hex'.format(name))
                    if sv:
                        self.verbose("saving file {}, shape: {}".format('cmodel/acc_just_after_layer_{}_and_before_add_bias.hex'.format(name), eta_conv.shape))
                    else:
                        self.verbose("saving file {} failed".format('cmodel/acc_just_after_layer_{}_and_before_add_bias.hex'.format(name)))

                    #done
                    self.args.global_buffer.pop('print-{}'.format(input_index))
                    self.args.global_buffer.pop('print-{}-weight'.format(input_index))

                eta_bias = eta_bias.reshape(1, -1, 1, 1)
                eta_norm = eta_conv + eta_bias
                #eta_norm = torch.clamp(eta_norm, min=-self.level_num//2, max=self.level_num-1-self.level_num//2)

                #if self.index in [88]:
                #    import pdb
                #    pdb.set_trace()
                output = eta_norm * input_scale
                alpha = alpha.reshape(1, -1, 1, 1)
                return output * alpha
            else:
                return super(BatchNorm2d, self).forward(x)
        else:
            raise RuntimeError("should not reach here")

    def __repr__(self):
        base = super(BatchNorm2d, self).__repr__()
        if self.enable:
            for item in self.items:
                base = base + "-{}({})".format(item, getattr(self, item))
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
        self.name = ""
        self.tag = 'split'
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
            base = base + "-first_half({})-index({})-input({})-name({})".format(self.first_half, self.index, self.input_index, self.name)
        return base

    def forward(self, x):
        N, C, H, W = x.shape
        splits = torch.chunk(x, 2, dim=1)
        y = splits[0] if self.first_half else splits[1]

        if self.enable and 'eval' in self.args.keyword:
            scaled = None
            if 'split-{}'.format(self.index) not in self.args.global_buffer:
                input_index = self.input_index.split('/')
                scale = []
                for i in input_index:
                    if i in self.args.global_buffer:
                        scale.append(self.args.global_buffer[i])
                    else:
                        self.verbose("warning {} not found in global_buffer, disable split layer-{}".format(
                            i, self.index))
                        self.enable = False

                if self.enable:
                    scaled = np.array(scale)
                    scaled = scaled.reshape(-1)
                    if self.first_half:
                        scaled = scaled[:C//2]
                    else:
                        scaled = scaled[C//2:]
                    self.args.global_buffer['split-{}'.format(self.index)] = scaled
                    #self.verbose("add split-{} to global_buffer".format(self.index))
            else:
                scaled = self.args.global_buffer['split-{}'.format(self.index)]

            if scaled is not None:
                if 'print' in self.args.keyword and self.name != "":
                    assert '/' not in self.input_index
                    scaled = scaled / 15.0
                    scaled = torch.from_numpy(scaled).to(device=y.device, dtype=y.dtype)
                    scaled = scaled.reshape(1, -1, 1, 1)
                    eta_z = y.div(scaled).round()
                    name = self.name + ('1' if self.first_half else '2')
                    sv = True
                    sv = st.tensor_to_txt(eta_z, 4, signed=False, filename='cmodel/output-fm-of-layer-{}.hex'.format(name))
                    if sv:
                       self.verbose("saving file {}, shape: {}".format('cmodel/output-fm-of-layer-{}.hex'.format(name), eta_z.shape))
                    else:
                       self.verbose("saving file {} failed".format('cmodel/output-fm-of-layer-{}.hex'.format(name)))

        return y

def split(dim, first_half, args=None):
    return Split(dim=dim, first_half=first_half, args=args)


class Concat(nn.Module):
    def __init__(self, args=None):
        super(Concat, self).__init__()
        self.index = -1
        self.verbose = print
        self.enable = False
        self.input_index = ""
        self.tag = 'concat'
        self.name = ""
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
            base = base + "-index({})-input({})-name({})".format(self.index, self.input_index, self.name)
        return base

    def forward(self, x, y):
        z = torch.cat((x, y), dim=1)

        if self.enable and 'eval' in self.args.keyword:
            scaled = None
            if 'concat-{}'.format(self.index) not in self.args.global_buffer:
                input_index = self.input_index.split('/')
                assert len(input_index) == 2
                def get_scale(x, input_index):
                    N, C, H, W = x.shape
                    if input_index in self.args.global_buffer:
                        part = self.args.global_buffer[input_index]
                        if isinstance(part, torch.Tensor):
                            part = part.cpu().numpy()
                        if len(part) == 1:
                            scale = [part[0] for i in range(C)]
                        elif len(part) == C:
                            scale = part.tolist()
                        else:
                            print("unexpected length of scale", len(part))
                            pdb.set_trace()
                    else:
                        self.verbose("warning {} not found in global_buffer, disable concat layer-{}".format(
                            i, self.index))
                        self.enable = False
                    return scale
                scale = get_scale(x, input_index[0])
                scale = scale + get_scale(y, input_index[1])

                if self.enable:
                    scaled = np.array(scale)
                    scaled = scaled.reshape(-1)
                    self.args.global_buffer['concat-{}'.format(self.index)] = scaled
                    #self.verbose("add concat-{} to global_buffer".format(self.index))
            else:
                scaled = self.args.global_buffer['concat-{}'.format(self.index)]

            if scaled is not None:
                if 'print' in self.args.keyword and self.name != "":
                    #pdb.set_trace()
                    scaled = scaled / 15.0
                    scaled = torch.from_numpy(scaled).to(device=y.device, dtype=y.dtype)
                    scaled = scaled.reshape(1, -1, 1, 1)
                    eta_z = z.div(scaled).round()
                    name = self.name
                    sv = True
                    sv = st.tensor_to_txt(eta_z, 4, signed=False, filename='cmodel/output-fm-of-layer-{}.hex'.format(name))
                    if sv:
                       self.verbose("saving file {}, shape: {}".format('cmodel/output-fm-of-layer-{}.hex'.format(name), eta_z.shape))
                    else:
                       self.verbose("saving file {} failed".format('cmodel/output-fm-of-layer-{}.hex'.format(name)))

        return z

def concat(args=None):
    return Concat(args=args)


