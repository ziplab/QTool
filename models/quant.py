
import math
import sys
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _quadruple

if sys.version_info[0] == 3:
    from . import alqnet as alqnet
    from . import dorefa as dorefa
    from . import xnor as xnor

__EPS__ = 0 #1e-5

class quantization(nn.Module):
    def __init__(self, args=None, tag='fm', shape=[], feature_stride=None, logger=None, groups=1):
        super(quantization, self).__init__()
        self.args = args
        self.logger = logger
        self.index = -1
        self.tag = tag
        self.method = 'none'
        self.choice = 'none'
        if logger is None:
            if hasattr(args, 'logger'):
                self.logger = args.logger
            else:
                logger_root = args.logger_root + '.' if hasattr(args, 'logger_root') else ''
                self.logger = logging.getLogger(logger_root + __name__)

        self.shape = shape
        self.feature_stride = feature_stride
        self.enable = getattr(args, tag + '_enable', False)
        self.adaptive = getattr(self.args, self.tag + '_adaptive', 'none')
        self.grad_scale = getattr(self.args, self.tag + '_grad_scale', 'none')
        self.grad_type = getattr(args, tag + '_grad_type', 'none')
        self.custom = getattr(args, tag + '_custom', 'none')
        self.bit = getattr(args, tag + '_bit', None)
        self.num_levels = getattr(args, tag + '_level', None)
        self.half_range = getattr(args, tag + '_half_range', None)
        self.scale = getattr(args, tag + '_scale', 0.5)
        self.ratio = getattr(args, tag + '_ratio', 1.0)
        self.correlate = getattr(args, tag + '_correlate', -1.0)
        self.quant_group = getattr(args, tag + '_quant_group', None)
        self.boundary = getattr(self.args, self.tag + '_boundary', None)
        if self.bit is None:
            self.bit = 32
        if self.num_levels is None or self.num_levels <= 0:
            self.num_levels = int(2 ** self.bit)
        self.bit = (int)(self.bit)
        if self.half_range is None:
            self.half_range = tag == 'fm'
        else:
            self.half_range = bool(self.half_range)

        self.grain = groups if 'grain' in getattr(self.args, 'keyword', []) else 1
        if self.quant_group == 0:
            self.quant_group = None
        if self.quant_group is not None:
            if self.quant_group < 0:
                if (shape[0] * shape[1]) % (-self.quant_group) != 0:
                    self.quant_group = None
                else:
                    self.quant_group = (shape[0] * shape[1]) / (-self.quant_group)
            else:
                if (shape[0] * shape[1]) % self.quant_group != 0:
                    self.quant_group = None
        if self.quant_group is not None:
            self.quant_group = int(self.quant_group)
        else:
            # layer wise for feature map, channel wise for weight
            self.quant_group = shape[0] if self.tag == 'wt' else 1
            ## channel wise for both
            #self.quant_group = shape[0] if self.tag == 'wt' else shape[1]
        self.norm_group = 1 if 'independent_norm' in getattr(self.args, 'keyword', []) else self.quant_group

        self.repeat_mark = 0
        self.input_index = ""

        if not self.enable:
            return

        self.fan = 1 # mode = 'fan_in' as default 
        for i in range(len(self.shape)-1):
            self.fan *= self.shape[i+1]

        self.nElements = 1
        for i in self.shape:
            self.nElements = self.nElements * i
        self.nElements = self.nElements // self.quant_group
        if self.tag in ['fm', 'ot']:
            batch_size = getattr(args, 'batch_size', 1)
            batch_size = getattr(args, 'batch_size_per_machine', batch_size)
            self.nElements *= batch_size
            if feature_stride is not None and hasattr(args, 'input_size') and args.input_size is not None:
                self.nElements *= (args.input_size // feature_stride)
                self.nElements *= (args.input_size // feature_stride)

        if 'proxquant' in getattr(self.args, 'keyword', []):
            self.prox = 0

        self.stable = getattr(args, self.tag + '_stable', 0)
        if self.stable <= 0:
            self.stable = getattr(args, 'stable', 0)

        self.iteration = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.level_num = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.adaptive_restore = False
        self.progressive = False
        self.quant_loss_enable = False
        self.quant_loss_function = 'None'
        self.quant_loss_alpha = 0.0
        self.init()
        self.level_num.fill_(self.num_levels)

        self.logger.info("half_range({}), bit({}), num_levels({}), quant_group({}) boundary({}) scale({}) ratio({}) tag({})".format(
            self.half_range, self.bit, self.num_levels, self.quant_group, self.boundary, self.scale, self.ratio, self.tag))
        if 'debug' in getattr(self.args, 'keyword', []):
            self.logger.info("adaptive({}) grad_scale({}) grad_type({}) norm_group({}) progressive({})".format(
                self.adaptive, self.grad_scale, self.grad_type, self.norm_group, self.progressive))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        string = "quantization-{}-index({})".format(self.tag, self.index)
        if self.args is not None and self.enable == True:
            string += "-enable({})-method({})-choice-({})-half_range({})-bit({})-quant_group({})-num_levels({})-level_num({})-adaptive({})".format(
                    self.enable, self.method, self.choice, self.half_range, self.bit, self.quant_group, self.num_levels, self.level_num.item(), self.adaptive)
        if self.input_index != "":
            string += "-input_index({})".format(self.input_index)
        return string

    def init(self):
        # for LQ-Net
        if 'lq' in self.args.keyword or 'alq' in self.args.keyword or 'popcount' in self.args.keyword:
            if not hasattr(self, 'num_levels'):
                self.num_levels = 2**self.bit
            if self.num_levels > 256:
                raise RuntimeError("currently not support more than 8 bit quantization")
            if self.num_levels == 3:
                self.bit = 1
                self.logger.info('update %s_bit %r' % (self.tag, self.bit))

            self.method = 'lqnet'
            if 'lq' in self.args.keyword:
                self.choice = 'lqnet'
            elif 'alq' in self.args.keyword:
                self.choice = 'alqnet'
            elif 'popcount' in self.args.keyword:
                self.choice = 'popcount'

            if 'lq' in self.args.keyword:
                self.lq_net_init()
                self.quant_fm = alqnet.LqNet_fm
                self.quant_wt = alqnet.LqNet_wt

            # initialize rould threshold
            init_thrs_multiplier = []
            for i in range(1, self.num_levels):
                thrs_multiplier_i = [0. for j in range(self.num_levels)]
                if not self.half_range:
                    if i < self.num_levels/2:
                        thrs_multiplier_i[i - 1] = 1 - self.scale
                        thrs_multiplier_i[i] = self.scale
                    elif i > self.num_levels/2:
                        thrs_multiplier_i[i - 1] = self.scale
                        thrs_multiplier_i[i] = 1 - self.scale
                    else:
                        thrs_multiplier_i[i - 1] = 0.5
                        thrs_multiplier_i[i] = 0.5
                else:
                    thrs_multiplier_i[i - 1] = self.scale
                    thrs_multiplier_i[i] = 1 - self.scale
                init_thrs_multiplier.append(thrs_multiplier_i)

            self.thrs_multiplier = nn.Parameter(torch.zeros(self.num_levels - 1, self.num_levels), requires_grad=False)
            self.thrs_multiplier.data = torch.FloatTensor(init_thrs_multiplier)
            if 'debug' in self.args.keyword:
                self.logger.info('self.thrs_multiplier: {}'.format(self.thrs_multiplier))

        if 'dorefa' in self.args.keyword or 'pact' in self.args.keyword:
            self.method = 'dorefa'
            self.gamma = 1.

            if self.boundary is None:
                self.boundary = 1.0
                self.logger.info('update %s_boundary %r' % (self.tag, self.boundary))
            self.grad_factor = {
                    'none': 1.0,
                    'fan-scale': np.sqrt(self.fan) * self.scale,
                    'scale-fan': self.scale / np.sqrt(self.fan),
                    'element-scale': np.sqrt(self.nElements) * self.scale,
                    'scale-element': self.scale / np.sqrt(self.nElements),
                    }[self.grad_scale]
            self.logger.info('update %s_grad_factor %f ( == 1 ? %s)' % 
                (self.tag, self.grad_factor, 'True' if self.grad_factor == 1 else 'False' ))

            if self.tag == 'fm':
                if 'lsq' in self.args.keyword or 'fm_lsq' in self.args.keyword:
                    if self.quant_group == 1:
                        self.clip_val = nn.Parameter(torch.Tensor([self.boundary]))
                    else:
                        self.clip_val = nn.Parameter(torch.zeros(1, self.quant_group, 1, 1))
                    self.clip_val.data.fill_(self.boundary)
                    self.quant = dorefa.LSQ
                    self.clamp = dorefa.ClampWithScale if self.grad_type in ['STE-scale'] else torch.clamp
                    self.choice = 'lsq'
                elif 'non-uniform' in self.args.keyword or 'fm_non-uniform' in self.args.keyword:
                    if self.quant_group == 1:
                        self.clip_val = nn.Parameter(torch.Tensor([self.boundary]), requires_grad = False)
                    else:
                        self.clip_val = nn.Parameter(torch.zeros(1, self.quant_group, 1, 1), requires_grad = False)
                    self.clip_val.data.fill_(self.boundary)
                    self.custom_ratio = self.ratio
                    self.quant = dorefa.RoundSTE
                    self.clamp = dorefa.ClampWithScale if self.grad_type in ['STE-scale'] else torch.clamp
                    assert self.num_levels <= 4, 'non-uniform target at 2bit, ter, bin'
                    assert self.half_range or self.num_levels == 3, 'Full range quantization for activation supports ternary only'
                    for i in range(self.num_levels-1):
                        if self.quant_group == 1:
                            setattr(self, "alpha%d" % i, nn.Parameter(torch.ones(1)))
                        else:
                            setattr(self, "alpha%d" % i, nn.Parameter(torch.ones(1, self.quant_group, 1, 1)))
                        getattr(self, "alpha%d" % i).data.fill_(self.scale / self.boundary)
                    self.choice = 'non-uniform'
                    if 'closed_form' in self.args.keyword or 'fm_closed_form' in self.args.keyword:
                        if self.quant_group != 1:
                            raise RuntimeError("function for verified")
                        self.basis = nn.Parameter(torch.ones (1), requires_grad=False)
                        self.auxil = nn.Parameter(torch.zeros(1), requires_grad=False)
                        self.choice = self.choice + '-with-closed_form'
                elif 'pact' in self.args.keyword:
                    self.quant = dorefa.qfn
                    self.clip_val = nn.Parameter(torch.Tensor([self.boundary]))
                    self.choice = 'pact'
                else: # Dorefa-Net
                    self.quant = dorefa.qfn
                    self.clip_val = self.boundary
                    self.choice = 'dorefa-net'
            elif self.tag == 'wt':
                if 'lsq' in self.args.keyword or 'wt_lsq' in self.args.keyword:
                    if self.shape[0] == 1 and self.quant_group != 1:  ## linear
                        raise RuntimeError("Quantization-{} for linear layer not provided".format(self.tag))
                    self.clip_val = nn.Parameter(torch.zeros(self.quant_group, 1, 1, 1))
                    self.clip_val.data.fill_(self.boundary)
                    self.quant = dorefa.LSQ
                    self.clamp = dorefa.ClampWithScale if self.grad_type in ['STE-scale'] else torch.clamp
                    assert self.half_range == False
                    self.choice = 'lsq'
                    if 'symmetry' in self.args.keyword:
                        assert self.bit > 1, "symmetry mode is only for bit greater than 1"
                        self.choice = self.choice + "-symmetry"
                        self.quant = dorefa.RoundSTE
                elif 'non-uniform' in self.args.keyword or 'wt_non-uniform' in self.args.keyword:
                    self.quant = dorefa.RoundSTE
                    self.clamp = dorefa.ClampWithScale if self.grad_type in ['STE-scale'] else torch.clamp
                    self.custom_ratio = self.ratio
                    assert self.num_levels == 3, 'non-uniform quantization for weight targets at ter'
                    for i in range(self.num_levels-1):
                        setattr(self, "alpha%d" % i, nn.Parameter(torch.ones(self.quant_group, 1, 1, 1)))
                        getattr(self, "alpha%d" % i).data.mul_(self.scale)
                    self.choice = 'non-uniform'
                elif 'wt_bin' in self.args.keyword and self.num_levels == 2:
                    self.quant = dorefa.DorefaParamsBinarizationSTE
                    self.choice = 'DorefaParamsBinarizationSTE'
                elif 'pact' in self.args.keyword:
                    self.quant = dorefa.qfn
                    self.clip_val = self.boundary
                    self.choice = 'dorefa-net'
                else:
                    self.choice = 'normalization'
                if 'gamma' in self.args.keyword or 'wt_gamma' in self.args.keyword:
                    if 'wt_gamma_in' in self.args.keyword:
                        self.gamma = np.sqrt(2 / (self.shape[1] // self.grain))
                    elif 'wt_gamma_out' in self.args.keyword:
                        self.gamma = np.sqrt(2 / (self.shape[0] // self.grain))
                    elif 'wt_gamma_learnable' in self.args.keyword:
                        self.gamma = nn.Parameter(torch.ones(self.quant_group, 1, 1, 1))
                        self.gamma.data.fill_(np.sqrt(2 / self.shape[0]))
                    self.choice = self.choice + '-with-gamma'
            elif self.tag == 'ot':
                if 'lsq' in self.args.keyword or 'ot_lsq' in self.args.keyword:
                    if self.quant_group == 1:
                        self.clip_val = nn.Parameter(torch.Tensor([self.boundary]))
                    else:
                        self.clip_val = nn.Parameter(torch.zeros(1, self.quant_group, 1, 1))
                    self.clip_val.data.fill_(self.boundary)
                    self.quant = dorefa.LSQ
                    self.clamp = dorefa.ClampWithScale if self.grad_type in ['STE-scale'] else torch.clamp
                    self.choice = 'lsq'
                elif 'non-uniform' in self.args.keyword or 'pact' in self.args.keyword:
                    raise RuntimeError("error keyword for the method, specific accurate tag please")
                else: # Dorefa-Net
                    self.quant = dorefa.qfn
                    self.clip_val = self.boundary
                    self.choice = 'dorefa-net'
                if 'gamma' in self.args.keyword or 'ot_gamma' in self.args.keyword:
                    self.gamma = nn.Parameter(torch.ones(1, self.quant_group, 1, 1))
                    self.choice = self.choice + '-with-gamma'
            else:
                raise RuntimeError("error tag for the method")


        if 'xnor' in self.args.keyword:
            self.method = 'xnor'
            if self.tag == 'fm':
                self.quant_fm = xnor.XnorActivation
                if 'debug' in self.args.keyword:
                    self.logger.info('debug: tag: {} custom: {}, grad_type {}'.format(self.tag, self.custom, self.grad_type))
                self.choice = 'xnor'
            elif self.tag == 'wt':
                if 'debug' in self.args.keyword:
                    self.logger.info('debug: tag: {} custom: {}, grad_type {}'.format(self.tag, self.custom, self.grad_type))
                self.quant_wt = xnor.XnorWeight
                self.choice = 'xnor'
                if 'gamma' in self.args.keyword:
                    self.gamma = nn.Parameter(torch.ones(self.quant_group, 1, 1, 1))
                    self.choice = 'xnor++'

        #raise RuntimeError("Quantization method not provided %s" % self.args.keyword)

    def update_quantization(self, **parameters):
        feedback = dict()
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
                    self.logger.warning('unexpect string in by_index: {}'.format(by_index))

            if by_index == 'all' or self.index in by_index:
                if ('by_tag' in parameters and self.tag in parameters['by_tag']) or ('by_tag' not in parameters):
                        for k, v in list(parameters.items()):
                            if hasattr(self, "{}".format(k)):
                                if isinstance(getattr(self, k), bool):
                                    v = False if v in ['False', 'false', False] else True
                                elif isinstance(getattr(self, k), int):
                                    v = int(v)
                                elif isinstance(getattr(self, k), float):
                                    v = float(v)
                                elif isinstance(getattr(self, k), str):
                                    v = v.replace("'", "").replace('"', '')
                                    if 'same' in v:
                                        v = v.replace('same', str(self.index))
                                    elif "last" in v:
                                        v = v.replace('last', str(self.index-1))
                                if isinstance(getattr(self, k), torch.Tensor):
                                    with torch.no_grad():
                                        if self.progressive:
                                            if 'lsq' in self.args.keyword or '{}_lsq'.format(self.tag) in self.args.keyword:
                                                if k in ['level_num']:
                                                    #if hasattr(self, 'clip_val'):
                                                    v = float(v)
                                                    v = v if v > 0 else self.level_num.item() + v # if negative number provide, it indicates decreasing on current
                                                    assert v > 1.9, "level_num should be at least 2"
                                                    scale = (v - 1) / (self.level_num.item() - 1)
                                                    self.clip_val.mul_(scale)
                                                    self.logger.info('update {}_clip_val to {} for index {}'.format(self.tag, self.clip_val, self.index))
                                                    # remember to patch the momentum in SGD optimizer. set it to zero or multiple by scale
                                                    if 'reset_momentum_list' in feedback:
                                                        feedback['reset_momentum_list'].append(self.clip_val)
                                                    else:
                                                        feedback['reset_momentum_list'] = [self.clip_val]
                                        getattr(self, k).fill_(float(v))
                                    self.logger.info('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))
                                else:
                                    setattr(self, "{}".format(k), v)
                                    self.logger.info('update {}_{} to {} for index {}'.format(self.tag, k, getattr(self, k, 'Non-Exist'), self.index))
                                if self.enable:
                                    assert hasattr(self, 'iteration'), "cannot enable quantization for current layer. Likely an error in policy file"
                            # global_buffer
                            if k in ['global_buffer']:
                                v = str(v)
                                if isinstance(getattr(self.args, k, None), dict) and hasattr(self, v) and self.enable:
                                    key = "{}-{}-{}".format(v, self.index, self.tag)
                                    self.args.global_buffer[key] = getattr(self, v)
                                    self.logger.info('update global_buffer (current length: {}), key: {}'.format(len(self.args.global_buffer), key))

        if not self.enable:
            return None
        else:
            if isinstance(self.quant_loss_function, str):
                if self.quant_loss_function == 'L2':
                    self.quant_loss_function = nn.MSELoss()
                elif self.quant_loss_function == 'L1':
                    self.quant_loss_function = nn.L1Loss()
                else:
                    self.quant_loss_function = 'none'
            assert self.method != 'none', "quantization enable but without specific method in layer(index:{}, tag:{})".format(self.index, self.tag)
            return feedback

    def init_based_on_warmup(self, data=None):
        if not self.enable:
            return

        with torch.no_grad():
            if self.method == 'dorefa' and data is not None:
                max_value = data.abs().max().item()
                if hasattr(self, 'clip_val') and isinstance(self.clip_val, torch.Tensor):
                    if self.correlate > 0:
                        max_value = max_value * self.correlate
                    self.clip_val.data = max_value + (self.iteration - 1) * self.clip_val.data
                    self.clip_val.div_(self.iteration.item())
                    #self.clip_val.fill_(max_value)
                    if self.iteration.data == self.stable:
                        self.logger.info('update %s clip_val for index %d to %r' % (self.tag, self.index, self.clip_val))
        return

    def init_based_on_pretrain(self, weight=None):
        if not self.enable:
            return

        with torch.no_grad():
            if self.method == 'dorefa' and 'non-uniform' in self.args.keyword:
                pass
        return

    def update_bias(self, basis=None):
        if not self.training:
            return

        if 'custom-update' not in self.args.keyword:
            self.basis.data = basis
            self.iteration.data = self.iteration.data + 1
        else:
            self.basis.data = self.basis.data * self.iteration  + self.auxil
            self.iteration.data = self.iteration.data + 1
            self.basis.data = self.basis.data / self.iteration

    def quantization_value(self, x, y):
        if self.iteration.data <= self.stable:
            self.init_based_on_warmup(x)
            return x
        elif 'proxquant' in self.args.keyword:
            return x * self.prox + y * (1 - self.prox)
        else:
            if 'probe' in self.args.keyword and self.index >= 0 and not self.training and self.tag == 'fm':
                for item in self.args.probe_list:
                    if 'before-quant' == item:
                        torch.save(x, "log/{}-activation-latent.pt".format(self.index))
                    elif 'after-quant' == item:
                        torch.save(y, "log/{}-activation-quant.pt".format(self.index))
                    elif hasattr(self, item):
                        torch.save(getattr(self, item), "log/{}-activation-{}.pt".format(self.index, item))
                self.index = -1
            if self.training and self.quant_loss_enable and isinstance(self.quant_loss_function, nn.Module):
                if 'quant_loss' in self.args.global_buffer:
                    self.args.global_buffer['quant_loss'] += self.quant_loss_function(x, y) * self.quant_loss_alpha
                else:
                    self.args.global_buffer['quant_loss'] = self.quant_loss_function(x, y) * self.quant_loss_alpha
            return y

    def coordinate(self, alpha):
        scale = [1.0/x if abs(x) < 1.0 else x for x in alpha] 

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
        scaled = [1.0/x if abs(alpha[i]) < 1.0 else x for i, x in enumerate(scaled)]
        return np.array(scaled)

    def forward(self, x):
        if not self.enable:
            return x

        if 'eval' in self.args.keyword and self.tag == 'fm' and 'skip' not in self.input_index:
            assert self.quant_group == 1 and self.method == 'dorefa' and self.half_range
            input_index_list = self.input_index.split('/')
            input_index = input_index_list[self.repeat_mark]
            if input_index in self.args.global_buffer:
                alpha = self.args.global_buffer[input_index]
                scaled = self.coordinate(alpha / self.clip_val.cpu().abs().item() * (self.level_num.item() - 1))
                scaled = scaled / alpha
                try:
                    scaled = torch.from_numpy(scaled).to(device=x.device, dtype=x.dtype).reshape(1, -1, 1, 1)
                except (ValueError, SyntaxError, TypeError) as e:
                    import pdb
                    pdb.set_trace()
                y = torch.round(x.mul(scaled))
                y = torch.clamp(y, max=(self.level_num.item() - 1))
                y = y.div((self.level_num.item() - 1) / self.clip_val)
                return y
            else:
                self.logger.warning("Integer only computation for layer {} - repeat mark {} might not supported.".format(self.index, self.repeat_mark))

        if self.method == 'lqnet':
            if self.tag == 'fm':
                y, basis = self.quant_fm.apply(x, self.basis, self.codec_vector, self.codec_index, self.thrs_multiplier, \
                        self.training, self.half_range, self.auxil, self.adaptive)
            else:
                y, basis = self.quant_wt.apply(x, self.basis, self.codec_vector, self.codec_index, self.thrs_multiplier, \
                        self.training, self.half_range, self.auxil, self.adaptive)

            self.update_bias(basis)

            return self.quantization_value(x, y)

        if 'xnor' in self.args.keyword:
            if self.tag == 'fm':
                y = self.quant_fm.apply(x, self.custom, self.grad_type)
            else:
                if self.adaptive == 'var-mean':
                    std, mean = torch.std_mean(x.data.reshape(self.norm_group, -1, 1, 1, 1), 1)
                    x = (x - mean) / (std + __EPS__)
                y = self.quant_wt.apply(x, self.quant_group, self.grad_type)
                if 'gamma' in self.args.keyword:
                    y = y * self.gamma

            return self.quantization_value(x, y)

        if self.method == 'dorefa':
            if self.tag in ['fm', 'ot']:
                if 'lsq' in self.args.keyword or '{}_lsq'.format(self.tag) in self.args.keyword:
                    clip_val = dorefa.GradientScale(self.clip_val.abs(), self.grad_factor)
                    if self.half_range:
                        y = x.div(clip_val)
                        y = self.clamp(y, min=0, max=1)
                        y = self.quant.apply(y, self.level_num.item() - 1)
                        y = y.mul(clip_val)
                    else:
                        y = x / clip_val
                        y = self.clamp(y, min=-1, max=1)
                        y = (y + 1.0) / 2.0
                        y = self.quant.apply(y, self.level_num.item() - 1)
                        y = y * 2.0 - 1.0
                        y = y * clip_val
                elif 'non-uniform' in self.args.keyword or '{}_non-uniform'.format(self.tag) in self.args.keyword:
                    b, c, h, w = x.shape
                    x = x.reshape(b, self.quant_group, 1, -1)
                    if self.half_range:
                        alpha0 = self.alpha0.abs()
                        y1 = x * alpha0
                        y1 = self.clamp(y1, min=0, max=1)
                        y1 = self.quant.apply(y1, self.custom_ratio)
                        y = y1
                        if self.num_levels >= 3:
                            alpha1 = self.alpha1.abs()
                            y2 = (x - 1.0/alpha0) * alpha1
                            y2 = self.clamp(y2, min=0, max=1)
                            y2 = self.quant.apply(y2, self.custom_ratio)
                            y = y + y2
                        if self.num_levels == 4:
                            alpha2 = self.alpha2.abs()
                            y3 = (x - (1.0/alpha0 + 1.0/alpha1)) * alpha2
                            y3 = self.clamp(y3, min=0, max=1)
                            y3 = self.quant.apply(y3, self.custom_ratio)
                            y =  y + y3
                    else:
                        alpha0 = self.alpha0.abs()
                        alpha1 = self.alpha1.abs()
                        y1 = x * alpha0
                        y1 = self.clamp(y1, min=-1, max=0)
                        y1 = self.quant.apply(y1, self.custom_ratio)
                        y2 = x * alpha1
                        y2 = self.clamp(y2, min=0, max=1)
                        y2 = self.quant.apply(y2, self.custom_ratio)
                        y = y1 + y2
                    if 'closed_form' in self.args.keyword or '{}_closed_form'.format(self.tag) in self.args.keyword:
                        if self.training:
                            self.auxil.data = dorefa.non_uniform_scale(x.detach(), y.detach())
                            self.update_bias(self.auxil.data)
                        y = y * self.basis
                    y = y.reshape(b, c, h, w)
                    x = x.reshape(b, c, h, w)
                elif 'pact' in self.args.keyword:
                    y = torch.clamp(x, min=0) # might not necessary when ReLU is applied in the network
                    y = torch.where(y < self.clip_val, y, self.clip_val)
                    y = self.quant.apply(y, self.num_levels, self.clip_val.detach(), self.adaptive)
                else: # default dorefa
                    y = torch.clamp(x, min=0, max=self.clip_val)
                    y = self.quant.apply(y, self.num_levels, self.clip_val, self.adaptive)
                if 'gamma' in self.args.keyword or '{}_gamma'.format(self.tag) in self.args.keyword:
                    y = y * self.gamma
            elif self.tag == 'wt':
                if self.adaptive == 'var-mean':
                    if hasattr(torch, 'std_mean'):
                        std, mean = torch.std_mean(x.data.reshape(self.norm_group, -1, 1, 1, 1), 1)
                    else:
                        _data = x.data.reshape(self.norm_group, -1, 1, 1, 1)
                        std = torch.std(_data, 1)
                        mean = torch.mean(_data, 1)
                    x = (x - mean) / (std + __EPS__)
                if 'lsq' in self.args.keyword or 'wt_lsq' in self.args.keyword:
                    clip_val = dorefa.GradientScale(self.clip_val.abs(), self.grad_factor)
                    c1, c2, kh, kw = x.shape
                    x = x.reshape(self.quant_group, -1, kh, kw)
                    if 'symmetry' in self.args.keyword:
                        y = x / clip_val * (self.level_num.item() // 2)
                        y = self.quant.apply(y)
                        y = torch.clamp(y, min=- self.level_num.item() // 2, max=self.level_num.item() - self.level_num.item() //2 - 1)
                        y = y / (self.level_num.item() // 2)  * clip_val
                    else:
                        y = x / clip_val
                        y = self.clamp(y, min=-1, max=1)
                        y = (y + 1.0) / 2.0
                        y = self.quant.apply(y, self.level_num.item() - 1)
                        y = y * 2.0 - 1.0
                        y = y * clip_val
                    y = y.reshape(c1, c2, kh, kw)
                    x = x.reshape(c1, c2, kh, kw)
                elif 'non-uniform' in self.args.keyword or 'wt_non-uniform' in self.args.keyword:
                    c1, c2, kh, kw = x.shape
                    x = x.reshape(self.quant_group, -1, kh, kw)
                    alpha0 = self.alpha0.abs()
                    alpha1 = self.alpha1.abs()
                    y1 = x * alpha0
                    y1 = self.clamp(y1, min=-1, max=0)
                    y1 = self.quant.apply(y1, self.custom_ratio)
                    y2 = x * alpha1
                    y2 = self.clamp(y2, min=0, max=1)
                    y2 = self.quant.apply(y2, self.custom_ratio)
                    y = y1 + y2
                    y = y.reshape(c1, c2, kh, kw)
                    x = x.reshape(c1, c2, kh, kw)
                elif 'wt_bin' in self.args.keyword and self.num_levels == 2:
                    y = self.quant.apply(x, self.adaptive)
                elif 'wt_dorefa' in self.args.keyword:
                    y = torch.tanh(x)
                    y = y / (2 * y.abs().max()) + 0.5
                    y = 2 * self.quant.apply(y, self.num_levels, self.clip_val, self.adaptive) - 1
                else:
                    y = x
                if 'gamma' in self.args.keyword or 'wt_gamma' in self.args.keyword:
                    y = y * self.gamma
                    x = x * self.gamma
                if self.adaptive_restore and self.adaptive == 'var-mean':
                    y = y * (std + __EPS__) + mean
            else:
                raise RuntimeError("Should not reach here for Dorefa-Net method")

            self.iteration.data = self.iteration.data + 1
            return self.quantization_value(x, y)

        raise RuntimeError("Should not reach here in quant.py")

    def lq_net_init(self):
        self.basis = nn.Parameter(torch.ones(self.bit, self.quant_group), requires_grad=False)
        self.auxil = nn.Parameter(torch.zeros(self.bit, self.quant_group), requires_grad=False)
        self.codec_vector = nn.Parameter(torch.ones(self.num_levels, self.bit), requires_grad=False)
        self.codec_index = nn.Parameter(torch.ones(self.num_levels, dtype=torch.int), requires_grad=False)

        init_basis = []
        NORM_PPF_0_75 = 0.6745
        if self.tag == 'fm':
            base = NORM_PPF_0_75 * 2. / (2 ** (self.bit - 1))
        elif self.tag == 'wt':
            base = NORM_PPF_0_75 * ((2. / self.fan) ** 0.5) / (2 ** (self.bit - 1))
        for i in range(self.bit):
            init_basis.append([(2 ** i) * base for j in range(self.quant_group)])
        self.basis.data = torch.FloatTensor(init_basis)

        # initialize level_codes
        init_level_multiplier = []
        for i in range(self.num_levels):
            level_multiplier_i = [0. for j in range(self.bit)]
            level_number = i
            for j in range(self.bit):
                binary_code = level_number % 2
                if binary_code == 0 and not self.half_range:
                    binary_code = -1
                level_multiplier_i[j] = float(binary_code)
                level_number = level_number // 2
            init_level_multiplier.append(level_multiplier_i)
        self.codec_vector.data = torch.FloatTensor(init_level_multiplier)

        init_codec_index = []
        for i in range(self.num_levels):
            init_codec_index.append(i)
        self.codec_index.data= torch.IntTensor(init_codec_index)

class custom_conv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
            args=None, force_fp=False, feature_stride=1, bits_weights=32, bits_activations=32):
        super(custom_conv, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.args = args
        self.force_fp = force_fp
        if not self.force_fp:
            self.pads = padding
            self.padding = (0, 0)
            self.quant_activation = quantization(args, 'fm', [1, in_channels, 1, 1], feature_stride=feature_stride, groups=groups)
            self.quant_weight = quantization(args, 'wt', [out_channels, in_channels, kernel_size, kernel_size], groups=groups)
            self.quant_output = quantization(args, 'ot', [1, out_channels, 1, 1])
            self.padding_after_quant = getattr(args, 'padding_after_quant', False) if args is not None else False
            assert self.padding_mode != 'circular', "padding_mode of circular is not supported yet"

    def init_after_load_pretrain(self):
        if not self.force_fp:
            self.quant_activation.init_based_on_pretrain()
            self.quant_weight.init_based_on_pretrain(self.weight.data)
            self.quant_output.init_based_on_pretrain()

    def update_quantization_parameter(self, **parameters):
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
            fd = self.quant_weight.update_quantization(**parameters)
            merge_dict(feedback, fd)
            fd = self.quant_output.update_quantization(**parameters)
            merge_dict(feedback, fd)
            return feedback
        else:
            return None

    def forward(self, inputs):
        if not self.force_fp:
            weight = self.quant_weight(self.weight)
            if self.padding_after_quant:
                inputs = self.quant_activation(inputs)
                inputs = F.pad(inputs, _quadruple(self.pads), 'constant', 0)
            else: # ensure the correct quantization levels (for example, BNNs only own the -1 and 1. zero-padding should be quantized into one of them
                inputs = F.pad(inputs, _quadruple(self.pads), 'constant', 0)
                inputs = self.quant_activation(inputs)
        else:
            weight = self.weight

        output = F.conv2d(inputs, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        if not self.force_fp:
            output = self.quant_output(output)

        return output

def conv5x5(in_planes, out_planes, stride=1, groups=1, padding=2, bias=False, args=None, force_fp=False, feature_stride=1, keepdim=True):
    "5x5 convolution with padding"
    return custom_conv(in_planes, out_planes, kernel_size=5, stride=stride, padding=padding, groups=groups, bias=bias,
            args=args, force_fp=force_fp, feature_stride=feature_stride)

def conv3x3(in_planes, out_planes, stride=1, groups=1, padding=1, bias=False, args=None, force_fp=False, feature_stride=1, keepdim=True):
    "3x3 convolution with padding"
    return custom_conv(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, groups=groups, bias=bias,
            args=args, force_fp=force_fp, feature_stride=feature_stride)

def conv1x1(in_planes, out_planes, stride=1, groups=1, padding=0, bias=False, args=None, force_fp=False, feature_stride=1, keepdim=True):
    "1x1 convolution"
    return custom_conv(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, groups=groups, bias=bias,
            args=args, force_fp=force_fp, feature_stride=feature_stride)

def conv0x0(in_planes, out_planes, stride=1, groups=1, padding=0, bias=False, args=None, force_fp=False, feature_stride=1, keepdim=True):
    "nop"
    return nn.Sequential()

def qconv(in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, args=None, force_fp=False, feature_stride=1, keepdim=True):
    return custom_conv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias,
            args=args, force_fp=force_fp, feature_stride=feature_stride)

class custom_linear(nn.Linear):
    def __init__(self, in_channels, out_channels, dropout=0, bias=True, args=None):
        super(custom_linear, self).__init__(in_channels, out_channels, bias=bias)
        self.args = args
        self.dropout = dropout
        self.force_fp = True
        if self.args is not None and hasattr(self.args, 'keyword'):
            self.quant_activation = quantization(args, 'fm', [1, in_channels, 1, 1])
            self.quant_weight = quantization(args, 'wt', [1, 1, out_channels, in_channels])
            self.force_fp = False

    #def init_after_load_pretrain(self):
    #    self.quant_weight.init_based_on_pretrain(self.weight.data)
    #    self.quant_activation.init_based_on_pretrain()

    def update_quantization_parameter(self, **parameters):
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
            fd = self.quant_weight.update_quantization(**parameters)
            return feedback
        else:
            return None

    def forward(self, inputs):
        if not self.force_fp:
            shape = self.weight.shape 
            weight = self.quant_weight(self.weight)
            weight = weight.reshape(shape)

            inputs = self.quant_activation(inputs)
        else:
            weight = self.weight

        output = F.linear(inputs, weight, self.bias)

        if self.dropout != 0:
            output = F.dropout(output, p=self.dropout, training=self.training)

        return output

def qlinear(in_planes, out_planes, dropout=0, bias=True, args=None):
    "1x1 convolution"
    return custom_linear(in_planes, out_planes, dropout=dropout, bias=bias, args=args)

class custom_eltwise(nn.Module):
    def __init__(self, channels=1, args=None, operator='sum'):
        super(custom_eltwise, self).__init__()
        self.args = args
        self.op = operator
        self.enable = False
        self.quant_x = None
        self.quant_y = None
        if hasattr(args, 'keyword') and hasattr(args, 'ot_enable') and args.ot_enable:
            self.enable = True
            if not args.ot_independent_parameter:
                self.quant = quantization(args, 'ot', [1, channels, 1, 1])
                self.quant_x = self.quant
                self.quant_y = self.quant
            else:
                raise RuntimeError("not fully implemented yet")

    def forward(self, x, y):
        z = None
        if self.op == 'sum':
            if self.enable:
                z = self.quant_x(x) + self.quant_y(y)
            else:
                z = x + y
        return z

    def update_quantization_parameter(self, **parameters):
        if self.enable:
            self.quant_x.update_quantization(**parameters)
            self.quant_y.update_quantization(**parameters)

def eltwise(channels=1, args=None, operator='sum'):
    return custom_eltwise(channels, args, operator)


