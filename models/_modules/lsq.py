"""
@inproceedings{
    esser2020learned,
    title={LEARNED STEP SIZE QUANTIZATION},
    author={Steven K. Esser and Jeffrey L. McKinstry and Deepika Bablani and Rathinakumar Appuswamy and Dharmendra S. Modha},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=rkgO66VKDS}
}
    https://quanoview.readthedocs.io/en/latest/_raw/LSQ.html
"""
import torch
import torch.nn.functional as F
import math
from models._modules import _Conv2dQ, Qmodes, _LinearQ
from enum import Enum

__all__ = ['Conv2dLSQ', 'LinearLSQ']



class ParametricQuantizer_kernelwise_decoupled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, alpha, beta, nbits, signed):
        ctx.nbits_preround = nbits
        nbits = torch.clamp(nbits,min =2)

        # nbits = torch.round(nbits)
        ctx.nbits = nbits
        if signed:
            Qp = torch.round(((2**(nbits-1))))-1
            Qn = torch.round((-2**(nbits-1)))
        else:
            Qp = torch.round(((2**(nbits)-1)))
            Qn = 0*torch.ones(nbits.shape).to(nbits.device)
        
        alpha = alpha.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        beta = beta.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        ctx.save_for_backward(v, alpha, beta, Qp,Qn)
        ctx.signed = signed
        vq = torch.clamp(torch.round(v/beta),Qn,Qp)*alpha
        
        return vq
    
    def backward(ctx, grad_output):
        v,alpha,beta, Qp,Qn = ctx.saved_tensors
        v_d = v/beta
        greater = torch.gt(torch.round(v_d),Qp)
        lesser = torch.lt(torch.round(v_d),Qn)
        signed = ctx.signed
        nbits = ctx.nbits.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        nbits_preround = ctx.nbits_preround
        
        #get gradients for input
        grad_input = grad_output.clone() * alpha/beta
        grad_input[torch.logical_or(greater,lesser)] = 0
        
        #get gradients for alpha
        grad_alpha = torch.round(v_d)
        grad_alpha[greater] = Qp
        grad_alpha[lesser] = Qn
        grad_alpha = grad_alpha / torch.sqrt(v[0].numel() * Qp)
        grad_alpha = torch.mul(grad_alpha,grad_output.clone())
        grad_alpha = torch.sum(grad_alpha, dim = (1,2,3))

        #get gradients for beta
        grad_beta = -1.0*(alpha/beta**2)*v
        grad_beta[greater] = 0
        grad_beta[lesser] = 0
        grad_beta = grad_beta / torch.sqrt(v[0].numel() * Qp)
        grad_beta = torch.mul(grad_beta, grad_output.clone())
        grad_beta= torch.sum(grad_beta, dim = (1,2,3))
        
        #get gradients for nbits
        grad_Qp = alpha * grad_output.clone()
        grad_Qp[torch.le(torch.round(v_d), Qp)] = 0
        grad_Qn = alpha * grad_output.clone()
        grad_Qn[torch.ge(torch.round(v_d), Qn)] = 0
        if signed:
            grad_nbits = (2**(nbits-1))*torch.log(torch.tensor([2]).to(nbits.device))*(grad_Qp - grad_Qn)
        else:
            grad_nbits = (2**(nbits))*torch.log(torch.tensor([2]).to(nbits.device))*(grad_Qp - grad_Qn)
        grad_nbits = torch.sum(grad_nbits).unsqueeze(0)
        grad_nbits[nbits_preround.le(2)] = 0
        
        return grad_input, grad_alpha, grad_beta, grad_nbits, None, None

    
class ParametricQuantizer_decoupled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, alpha, beta, nbits, signed):
        
        ctx.nbits_preround = nbits
        nbits = torch.clamp(nbits,min =2)
        # nbits = torch.round(nbits)
        ctx.nbits = nbits
        if signed:
            Qp = torch.round(((2**(nbits-1))))-1
            Qn = torch.round((-2**(nbits-1)))
        else:
            Qp = torch.round(((2**(nbits)-1)))
            Qn = 0*torch.ones(nbits.shape).to(nbits.device)
        
        ctx.save_for_backward(v, alpha, beta, Qp,Qn)
        ctx.signed = signed
        vq = torch.clamp(torch.round(v/beta),Qn,Qp)*alpha
        
        
        return vq
    
    def backward(ctx, grad_output):
        v,alpha,beta,Qp,Qn = ctx.saved_tensors
        nbits = ctx.nbits
        nbits_preround = ctx.nbits_preround
        signed = ctx.signed
        v_d = v/beta
        greater = torch.gt(torch.round(v_d),Qp)
        lesser = torch.lt(torch.round(v_d),Qn)
        
        #get gradients for input
        grad_input = grad_output.clone() * alpha/beta
        grad_input[torch.logical_or(greater,lesser)] = 0
        
        #get gradients for alpha
        grad_alpha = torch.round(v_d)
        grad_alpha[greater] = Qp
        grad_alpha[lesser] = Qn
        grad_alpha = grad_alpha / torch.sqrt(v[0].numel() * Qp)
        grad_alpha = torch.mul(grad_alpha,grad_output.clone())
        grad_alpha = torch.sum(grad_alpha).unsqueeze(0)
        
        #get gradients for beta
        grad_beta = -1.0 * (alpha/beta**2) * v
        grad_beta[greater] = 0
        grad_beta[lesser] = 0
        grad_beta = grad_beta / torch.sqrt(v[0].numel() * Qp)
        grad_beta = torch.mul(grad_beta,grad_output.clone())
        grad_beta = torch.sum(grad_beta).unsqueeze(0)

        #get gradients for nbits
        grad_Qp = alpha * grad_output.clone()
        grad_Qp[torch.le(torch.round(v_d), Qp)] = 0
        grad_Qn = alpha * grad_output.clone()
        grad_Qn[torch.ge(torch.round(v_d), Qn)] = 0
        if signed:
            grad_nbits = (2**(nbits-1))*torch.log(torch.tensor([2]).to(nbits.device))*(grad_Qp - grad_Qn)
        else:
            grad_nbits = (2**(nbits))*torch.log(torch.tensor([2]).to(nbits.device))*(grad_Qp - grad_Qn)
        grad_nbits = torch.sum(grad_nbits).unsqueeze(0)
        grad_nbits[nbits_preround.le(2)] = 0
        return grad_input, grad_alpha, grad_beta, grad_nbits, None, None










def floor_pass(x):
    y = x.floor()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad

def sigmoid_backward(v):
    return torch.sigmoid(v)*(1-torch.sigmoid(v))

def round_sigmoid(x, T):
    y = x.round()
    y_grad = torch.sigmoid((x - 0.5*(x.floor()+x.ceil()))/T) + x.floor()
    return y.detach() - y_grad.detach() + y_grad





def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return y.detach() - y_grad.detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return y.detach() - y_grad.detach() + y_grad


class Conv2dLSQ(_Conv2dQ):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, nbits_w=8, nbits_a=8, qmode='layer_wise', **kwargs):
        super(Conv2dLSQ, self).__init__(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias,
            nbits_w=nbits_w,nbits_a=nbits_a, qmode = qmode)

    def forward(self, x):
        
        if self.training and self.init_state == 0:
            Qp_w = (2 ** (self.bits_w - 1)-1).int()
            Qn_w = (-2 ** (self.bits_w - 1)).int()
            self.alpha_w.data.copy_(2 * torch.mean(self.weight.abs(), dim = (1,2,3), keepdim = False) / torch.sqrt(Qp_w))
            self.beta_w.data.copy_(2 * torch.mean(self.weight.abs(), dim = (1,2,3), keepdim = False) / torch.sqrt(Qp_w))
            
            self.bits_w = self.bits_w.to(self.weight.device)

            
            
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qp_a = 2 ** (self.bits_a - 1) -1
                Qn_a = -2**(self.bits_a - 1)
            else:
                Qp_a = 2 ** self.bits_a - 1
                Qn_a = 0
            self.alpha_a.data.copy_(2*x.abs().mean() / math.sqrt(Qp_a))
            self.beta_a.data.copy_(2*x.abs().mean() / math.sqrt(Qp_a))
            self.bits_a = self.bits_a.to(x.device)
            self.num_elements_w.fill_(self.weight.numel())
            self.num_elements_a.fill_(x[0].numel())
        
                
        #Quantize weights
        weight = self.weight
        w_q = ParametricQuantizer_kernelwise_decoupled.apply(weight,self.alpha_w,2*self.beta_w, self.bits_w,True)



        #Quantize Activations
        x_q = ParametricQuantizer_decoupled.apply(x,self.alpha_a, 2*self.beta_a, self.bits_a,self.signed)


        output = F.conv2d(x_q, w_q, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        if self.training and self.init_state == 0:
            #bitops = nbits_a * nbits_w * OFM_H * OFM_W * K * C * kx * ky
            #nbits multiplied in train() function
            self.bitops.fill_(output.size()[1] * output.size()[2] * output.size()[3] * self.in_channels * self.kernel_size[0] * self.kernel_size[1] / self.groups)
            self.init_state.fill_(1)
        return output


class LinearLSQ(_LinearQ):
    def __init__(self, in_features, out_features, bias=True, nbits_w=4, **kwargs):
        super(LinearLSQ, self).__init__(in_features=in_features,
                                        out_features=out_features, bias=bias, nbits=nbits_w)

    def forward(self, x):
        # if self.alpha is None:
        #     return F.linear(x, self.weight, self.bias)
        if self.training and self.init_state == 0:
            Qp_w = 2 ** (self.bits_w - 1)
            self.alpha_w.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp_w))
            self.beta_w.data.copy_(2 * self.weight.abs().mean() / math.sqrt(Qp_w))
            self.bits_w = self.bits_w.to(self.weight.device)
            if x.min() < -1e-5:
                self.signed.data.fill_(1)
            if self.signed == 1:
                Qp_a = 2 ** (self.bits_a - 1) 
            else:
                Qp_a = 2 ** self.bits_a - 1
            self.alpha_a.data.copy_(2 * x.abs().mean() / math.sqrt(Qp_a))
            self.beta_a.data.copy_(2 * x.abs().mean() / math.sqrt(Qp_a))

            self.bits_a = self.bits_a.to(x.device)
            self.num_elements_w.fill_(self.weight.numel())
            self.num_elements_a.fill_(x[0].numel())
            self.bitops.fill_(x.size()[1] * self.out_features)
            self.temperature = self.temperature.to(self.weight.device)
            self.init_state.fill_(1)
        
        #Quantize Activations
        x_q = ParametricQuantizer_decoupled.apply(x,self.alpha_a,self.beta_a, self.bits_a,self.signed)
        #Quantize weights
        w_q = ParametricQuantizer_decoupled.apply(self.weight,self.alpha_w,self.beta_w, self.bits_w,True)

        return F.linear(x_q, w_q, self.bias)


