import torch
import torch.nn as nn
import torch.nn.init as init

# https://arxiv.org/pdf/1606.03498.pdf
class MinibatchDiscriminator(nn.Module):
    def __init__(self, in_features: int, out_features: int, kernel_dims: int):
        super(MinibatchDiscriminator, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_dims = kernel_dims

        # Where A is the number of input features, B is the number of output features, and C is the number of kernel dimensions.
        # T = R^(AxBxC)
        # To allow T to be multiplied by R^(A), B and C will be multiplied to form a matrix of size AxBC
        self.T = nn.Parameter(torch.Tensor(in_features, out_features * kernel_dims))
        # Initialize with gaussian values
        init.normal_(self.T, 0, 1)

    def forward(self, x: torch.Tensor):
        # f(xi) = R^(A)
        # therefore f(x) = R^(NxA)
        # x is of shape N by A (where A is the input feature count)

        # Mi = R^(BxC)
        # therefore M = R^(NxBxC)
        matrices = x.mm(self.T) # Nx(BC)
        # expand out to NxBxC
        matrices = matrices.view(-1, self.out_features, self.kernel_dims)

        # Insert extra dimension and rearrange a copy so the matrices can be used in broadcasting
        # broadcasting is where the tensor is automatically expanded to match the shape of the other tensor without copying the data
        Mib = matrices.unsqueeze(0) # 1xNxBxC
        Mjb = Mib.permute(1, 0, 2, 3) # Nx1xBxC

        # Use automatic broadcasting to perform efficient calculate of L1 distance
        # c(xi, xj) = exp( -|| Mib - Mjb || )
        norm = torch.abs(Mib - Mjb).sum(3)
        # o(X) = R^(NxB)
        oX = torch.exp(-norm).sum(0)

        x = torch.cat((x, oX), 1)
        return x