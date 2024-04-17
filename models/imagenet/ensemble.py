import torch
import torch.nn as nn









class GeometricWeightedEnsemble(nn.Module):
    def __init__(self, num_heads: int, num_classes: int, base_model: nn.Module, exit_id: int):
        super().__init__()
        self.base_model = base_model
        self._num_heads = num_heads
        self._num_classes = num_classes
        self._weight = nn.Parameter(torch.normal(0, 0.01, size=(1, num_heads, 1)))
        self.exit_id = exit_id
        # self._weight = nn.Parameter(torch.zeros(size=(1, num_heads, 1)), requires_grad=  True)
        # self._weight.data[:,-1,:].copy_(1.0 * torch.tensor(1))
        self._bias = nn.Parameter(torch.ones(size=(1, num_classes,)), requires_grad = True)
        self.EPS = 1e-40
        # self._bias = nn.Parameter(torch.normal(0, 0.01, size=(1, num_classes,)))

    def forward(self, x):
        #x is dataset inputs
        with torch.no_grad():
            output_exit_student = self.base_model(x)
        output_exit_student = torch.stack(output_exit_student, dim = 1)
        x = output_exit_student[:, :(self.exit_id+1), :]

        # x are logits from the exits and the final classifier
        x = torch.log_softmax(x, dim = -1)
        # x shape is (batch_size, num_heads, num_classes)
        
        # x = torch.mean(x * weight, dim=1) + self._bias
        weight = self._weight.to(x.device)
        bias = self._bias.to(x.device)
        x = (((x * weight).sum(dim=1)).exp())*bias
        x = x/x.sum(dim=1, keepdim = True)
        #returns probabilities
        # x_log = (x + self.EPS).log()
        return x


class ArithmeticWeightedEnsemble(nn.Module):
    EPS = 1e-40

    def __init__(self, num_heads: int, num_classes: int):
        super().__init__()
        self._num_heads = num_heads
        self._num_classes = num_classes
        # self._weight = nn.Parameter(torch.ones(size=(1, num_heads, 1)), requires_grad=  True)
        self._weight = nn.Parameter(torch.normal(0, 0.01, size=(1, num_heads, 1)))
        self._bias = nn.Parameter(torch.zeros(size=(1, num_classes,)))

    def forward(self, x: torch.Tensor):
        x = torch.log_softmax(x, dim = -1)
        # print("X: ",x.shape)
        # print("W: ",self._weight.shape)
        # x are logprobs from the heads
        # x shape is (batch_size, num_heads, num_classes))
        x = x.exp()
        x = (x * self._weight.exp()).sum(dim=1) + self._bias.exp()
        x = x / x.sum(dim=1, keepdim=True)
        # back into logspace
        x = (x + self.EPS).log()
        return x