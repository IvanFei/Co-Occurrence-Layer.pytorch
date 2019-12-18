
import torch
import numpy as np
import matplotlib.pyplot as plt


def vis_matrix(matrix: torch.Tensor, input_shape: list, title: str, save_or_not: bool = False) -> None:
    matrix = matrix.reshape(input_shape).data.numpy()
    print(f"[*] matrix shape: {matrix.shape}")
    fig, ax = plt.subplots()
    im = ax.imshow(matrix)
    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            text = ax.text(j, i, np.round(matrix[i, j], 2),
                           ha="center", va="center", color="w")
    ax.set_title(title)
    fig.tight_layout()
    if save_or_not:
        plt.savefig("../docs/imgs/" + title + ".png")
    plt.show()
