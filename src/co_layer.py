import torch
import math
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init


class CoOccurrenceLayer(nn.Module):
    def __init__(self, co_matrix_shape: list, spatial_shape: list, stride: int = 1) -> None:
        super(CoOccurrenceLayer, self).__init__()
        self.co_matrix_shape = co_matrix_shape
        self.spatial_shape = spatial_shape
        self.stride = stride
        self.co_matrix = Parameter(torch.Tensor(*self.co_matrix_shape))
        self.spatial_filter = Parameter(torch.Tensor(*self.spatial_shape))
        self.filters_ones = torch.ones([1, 1, 1, 1, 1], requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.co_matrix, a=math.sqrt(5))
        init.kaiming_uniform_(self.spatial_filter, a=math.sqrt(5))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        num_quantization = self.co_matrix_shape[0]
        input_idx = self._quantization_input_as_bins(input, num_quantization)
        conv_out = self._calculate_co_occurrence_conv(input, input_idx, num_quantization)
        return conv_out

    def _calculate_co_occurrence_conv(self, input: torch.Tensor, input_idx: torch.Tensor,
                                      num_quantizaiton: int) -> torch.Tensor:
        # for loop to calculate the conv
        N, c, h, w = input.shape
        input_idx_vec = input_idx.flatten()
        conv_out = torch.zeros_like(input)
        for i in range(num_quantizaiton):
            ith_row_co_matrix = self.co_matrix[i, :]
            # index_select
            cof_matrix = ith_row_co_matrix.index_select(dim=0, index=input_idx_vec.long())
            cof_matrix = cof_matrix.reshape([N, c, h, w])

            input_multiply_cof = cof_matrix * input
            input_multiply_cof = input_multiply_cof.unsqueeze(dim=1)
            # TODO the shape of w is consideration
            w_3d = self.spatial_filter.reshape([1, 1, *self.spatial_shape])
            input_mask = (input_idx == i).unsqueeze(dim=1).float()

            if input_mask.is_cuda:
                self.filters_ones = self.filters_ones.cuda()
            input_mask = torch.conv3d(input_mask, self.filters_ones, stride=[self.stride, self.stride, self.stride])
            padding = [int((self.spatial_shape[i] - 1) / 2) for i in range(len(self.spatial_shape))]
            spatial_conv_input = torch.conv3d(input_multiply_cof, w_3d,
                                              stride=[self.stride, self.stride, self.stride], padding=padding)

            conv_out += input_mask[:, 0, :, :, :] * spatial_conv_input[:, 0, :, :, :]

        return conv_out

    @staticmethod
    def _quantization_input_as_bins(input: torch.Tensor, num_quantization: int) -> torch.Tensor:
        input_min = torch.min(input).expand_as(input)
        input_max = torch.max(input).expand_as(input)
        # normalize the input to 0~1
        input_norm = (input - input_min) / input_max
        # ----- use round method -------- #
        # May cause the problem describe in issue #2
        # input to index
        # input_idx = input_norm * (num_quantization - 1)
        # # floor to int
        # input_idx = torch.round(input_idx).int()
        # ------ use floor method v2 -------- #
        eps = np.finfo(np.float32).eps
        input_idx = input_norm * num_quantization
        input_idx = np.abs(input_idx - eps)
        input_idx = np.floor(input_idx)

        return input_idx


if __name__ == "__main__":
    co_layer = CoOccurrenceLayer(co_matrix_shape=[5, 5], spatial_shape=[1, 3, 3])
    feat_map = np.random.rand(1, 1, 20, 10)
    print(f"[*] feature map shape: {feat_map.shape}")
    feat_map = torch.Tensor(feat_map)
    out = co_layer(feat_map)



