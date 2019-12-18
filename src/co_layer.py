import torch
import math
import numpy as np
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn import init

from typing import Tuple

from utils.utils import vis_matrix

Tensor_List = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


class CoOccurrenceLayer(nn.Module):
    def __init__(self, co_matrix_shape: list, spatial_shape: list, stride=1) -> None:
        super(CoOccurrenceLayer, self).__init__()
        self.co_matrix_shape = co_matrix_shape
        self.spatial_shape = spatial_shape
        self.stride = stride
        self.co_matrix = Parameter(torch.Tensor(*self.co_matrix_shape))
        self.spatial_filter = Parameter(torch.Tensor(*self.spatial_shape))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.co_matrix, a=math.sqrt(5))
        init.kaiming_uniform_(self.spatial_filter, a=math.sqrt(5))
        # init.constant_(self.spatial_filter, val=1)

    def forward(self, input: torch.Tensor) -> Tensor_List:
        num_quantization = self.co_matrix_shape[0]
        input_idx = self._quantization_input_as_bins(input, num_quantization)
        print(f"[*] input idx shape: {input_idx.shape}")
        conv_out = self._calculate_co_occurrence_conv(input, input_idx, num_quantization)
        return conv_out, self.co_matrix, self.spatial_filter, input_idx

    def _calculate_co_occurrence_conv(self, input: torch.Tensor, input_idx: torch.Tensor,
                                      num_quantizaiton: int) -> torch.Tensor:
        # for loop to calculate the conv
        N, c, h, w = input.shape
        input_idx_vec = input_idx.flatten()
        print(f"[*] input idx vec shape: {input_idx_vec.shape}")
        conv_out = torch.zeros_like(input)
        for i in range(num_quantizaiton):
            ith_row_co_matrix = self.co_matrix[i, :]
            # index_select
            print(f"[*] {i} th row matrix: {ith_row_co_matrix}")
            cof_matrix = ith_row_co_matrix.index_select(dim=0, index=input_idx_vec.long())
            print(f"[*] cof matrix shape: {cof_matrix.shape}")
            cof_matrix = cof_matrix.reshape([N, c, h, w])
            input_multiply_cof = cof_matrix * input
            input_multiply_cof = input_multiply_cof.unsqueeze(dim=1)
            # TODO the shape of w is consideration
            w_3d = self.spatial_filter.reshape([1, 1, *self.spatial_shape])
            input_mask = (input_idx == i).unsqueeze(dim=1).float()

            filters_ones = torch.ones([1, 1, 1, 1, 1])
            input_mask = torch.conv3d(input_mask, filters_ones, stride=[self.stride, self.stride, self.stride])
            padding = int((self.spatial_shape[0] - 1) / 2)
            spatial_conv_input = torch.conv3d(input_multiply_cof, w_3d,
                                              stride=[self.stride, self.stride, self.stride], padding=padding)

            print(f"[*] input_mask 0 shape: {input_mask[:, 0, :, :, :].shape}")
            print(f"[*] spatial_conv_input 0 shape: {spatial_conv_input[:, 0, :, :, :].shape}")
            conv_out += input_mask[:, 0, :, :, :] * spatial_conv_input[:, 0, :, :, :]

        return conv_out

    def _quantization_input_as_bins(self, input: torch.Tensor, num_quantization: int) -> torch.Tensor:
        input_min = torch.min(input).expand_as(input)
        input_max = torch.max(input).expand_as(input)
        # normalize the input to 0~1
        input_norm = (input - input_min) / input_max
        # input to index
        input_idx = input_norm * num_quantization
        # floor to int
        input_idx = torch.floor(input_idx).int()
        print(f"[*] input idx type: {type(input_idx)}")

        return input_idx


if __name__ == "__main__":
    co_layer = CoOccurrenceLayer(co_matrix_shape=[5, 5], spatial_shape=[3, 3, 3])
    feat_map = np.random.rand(1, 10, 10, 1)
    print(f"[*] feature map shape: {feat_map.shape}")
    feat_map = torch.Tensor(feat_map)
    out, co_matrix, spatial_filter, input_idx = co_layer(feat_map)

    vis_matrix(feat_map, [10, 10], "input matrix")

    vis_matrix(co_matrix, [5, 5], "co occurrence matrix")

    vis_matrix(input_idx, [10, 10], "input index")

    vis_matrix(out, [10, 10], "Co Occurrence output")
