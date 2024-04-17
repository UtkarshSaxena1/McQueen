"""
    Quantized modules: the base class
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import math
from enum import Enum

__all__ = ['Qmodes',  '_Conv2dQ', '_LinearQ',
           'truncation', 'round_pass', 'grad_scale']


class Qmodes(Enum):
    layer_wise = 1
    kernel_wise = 2


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad




def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad



def log_shift(value_fp):
    value_shift = 2 ** (torch.log2(value_fp).ceil())
    return value_shift


def clamp(input, min, max, inplace=False):
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def get_quantized_range(num_bits, signed=True):
    if signed:
        n = 2 ** (num_bits - 1)
        return -n, n - 1
    return 0, 2 ** num_bits - 1


def linear_quantize(input, scale_factor, inplace=False):
    if inplace:
        input.mul_(scale_factor).round_()
        return input
    return torch.round(scale_factor * input)


def linear_quantize_clamp(input, scale_factor, clamp_min, clamp_max, inplace=False):
    output = linear_quantize(input, scale_factor, inplace)
    return clamp(output, clamp_min, clamp_max, inplace)


def linear_dequantize(input, scale_factor, inplace=False):
    if inplace:
        input.div_(scale_factor)
        return input
    return input / scale_factor


def truncation(fp_data, nbits=8):
    il = torch.log2(torch.max(fp_data.max(), fp_data.min().abs())) + 1
    il = math.ceil(il - 1e-5)
    qcode = nbits - il
    scale_factor = 2 ** qcode
    clamp_min, clamp_max = get_quantized_range(nbits, signed=True)
    q_data = linear_quantize_clamp(fp_data, scale_factor, clamp_min, clamp_max)
    q_data = linear_dequantize(q_data, scale_factor)
    return q_data, qcode


def get_default_kwargs_q(kwargs_q, layer_type):
    default = {
        'nbits': 4
    }
    if isinstance(layer_type, _Conv2dQ):
        # default.update({
        #     'mode': Qmodes.layer_wise})
        pass
    elif isinstance(layer_type, _LinearQ):
        pass
    else:
        assert NotImplementedError
        return
    for k, v in default.items():
        if k not in kwargs_q:
            kwargs_q[k] = v
    return kwargs_q


class _Conv2dQ(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, nbits_a=8, qmode = 'layer_wise', **kwargs_q):
        super(_Conv2dQ, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
                                       padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        if qmode == 2:
            self.q_mode = Qmodes.kernel_wise
        else:
            self.q_mode = Qmodes.layer_wise
        quant_requires_grad = True
        self.alpha_w = Parameter(torch.Tensor(out_channels), requires_grad = quant_requires_grad)
        self.beta_w = Parameter(torch.Tensor(out_channels), requires_grad = quant_requires_grad)
        self.bits_w = Parameter(nbits_w * torch.ones(1), requires_grad = quant_requires_grad)
        self.alpha_a = Parameter(torch.Tensor(1), requires_grad = quant_requires_grad)
        self.beta_a = Parameter(torch.Tensor(1), requires_grad = quant_requires_grad)
        self.bits_a = Parameter(nbits_a * torch.ones(1), requires_grad = quant_requires_grad)
            
        
        self.register_buffer('signed', torch.zeros(1))
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('bitops', torch.zeros(1))
        self.register_buffer('num_elements_a', torch.zeros(1))
        self.register_buffer('num_elements_w', torch.zeros(1))

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def set_bit(self, nbits):
        self.bits_a = Parameter(nbits * torch.ones(1), requires_grad = False)


    def extra_repr(self):
        s_prefix = super(_Conv2dQ, self).extra_repr()
        return '{}, nbits_w:{}, nbits_a:{}'.format(s_prefix,self.bits_w.item(), self.bits_a[0].item() )



    
class _LinearQ(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,nbits_w=8, nbits_a=8, **kwargs_q):
        super(_LinearQ, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.kwargs_q = get_default_kwargs_q(kwargs_q, layer_type=self)
        # self.nbits = kwargs_q['nbits']
        
        self.alpha_a = Parameter(torch.Tensor(1))
        self.beta_a = Parameter(torch.Tensor(1), requires_grad = True)
        self.alpha_w = Parameter(torch.Tensor(1))
        self.beta_w = Parameter(torch.Tensor(1), requires_grad = True)
        self.bits_w = Parameter(nbits_w * torch.ones(1), requires_grad = True)
        self.bits_a = Parameter(nbits_a * torch.ones(1), requires_grad = True)
        self.register_buffer('init_state', torch.zeros(1))
        self.register_buffer('num_elements_w', torch.zeros(1))
        self.register_buffer('num_elements_a', torch.zeros(1))
        self.register_buffer('bitops', torch.zeros(1))
        self.register_buffer('signed', torch.zeros(1))
        self.temperature = torch.ones(1).to(self.weight.device)

    def add_param(self, param_k, param_v):
        self.kwargs_q[param_k] = param_v

    def extra_repr(self):
        s_prefix = super(_LinearQ, self).extra_repr()
        # if self.alpha_w is None:
        #     return '{}, fake'.format(s_prefix)
        return '{}, nbits_w:{}, nbits_a:{}'.format(s_prefix,self.bits_w.item(), self.bits_a.item() )

