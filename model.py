import torch
import numpy as np
from torch import nn

from dataset import ToyData
from src.co_layer import CoOccurrenceLayer

num_classification = 2


class CoOccurrenceNet(nn.Module):

    co_matrix_shape = [4, 4]
    w_shape = [10, 10, 1]
    conn_pool_size = (6, 6)

    def __init__(self) -> None:
        super(CoOccurrenceNet, self).__init__()
        self.conn = nn.Sequential(CoOccurrenceLayer(self.co_matrix_shape, self.w_shape),
                                  nn.ReLU())
        self.global_avgpooling = nn.AdaptiveAvgPool2d(self.conn_pool_size)
        self.classifier = nn.Linear(self.conn_pool_size[0] * self.conn_pool_size[1], num_classification)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(dim=1)
        input = input.reshape([input.shape[0], input.shape[1], ToyData.IMAGE_SIZE, ToyData.IMAGE_SIZE])
        conn_out = self.conn(input)
        pooling = self.global_avgpooling(conn_out)
        out = torch.flatten(pooling, 1)
        out = self.classifier(out)
        out = torch.softmax(out)

        return out


class FullConnectedNet(nn.Module):

    out_features = 36

    def __init__(self) -> None:
        super(FullConnectedNet, self).__init__()
        in_features = ToyData.IMAGE_SIZE * ToyData.IMAGE_SIZE
        self.linear = nn.Linear(in_features, self.out_features)
        self.classifier = nn.Linear(self.out_features, num_classification)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        features = self.linear(input)
        features = torch.sigmoid(features)
        out = self.classifier(features)
        out = torch.softmax(out, dim=1)

        return out


class ConvolutionNet(nn.Module):

    in_channels = 1
    out_channels = 36

    def __init__(self) -> None:
        super(ConvolutionNet, self).__init__()
        self.conv_layer = nn.Sequential(nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3),
                                        nn.ReLU(inplace=True))
        self.global_avgpooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(self.out_channels, num_classification)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.unsqueeze(dim=1)
        input = input.reshape([input.shape[0], input.shape[1], ToyData.IMAGE_SIZE, ToyData.IMAGE_SIZE])
        conv_out = self.conv_layer(input)
        avg_pool = self.global_avgpooling(conv_out)
        out = self.classifier(avg_pool)
        out = torch.flatten(out, 1)
        out = torch.softmax(out, dim=1)

        return out
