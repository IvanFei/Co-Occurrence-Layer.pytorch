import os
import torch
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorboardX as tb
from tensorboardX.summary import Summary
from argparse import Namespace

logger = logging.getLogger(__name__)

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


def prepare_dirs(args: Namespace) -> None:
    if args.model_name:
        if os.path.exists(os.path.join(args.log_dir, args.model_name)):
            raise FileExistsError(f"Model {args.model_name} already exists!! give a different name.")
    else:
        if args.load_path:
            if args.load_path.endswith(".pth"):
                args.model_dir = args.load_path.rsplit("/", 1)[0]
            else:
                args.model_dir = args.load_path
        else:
            raise Exception(f"At least one of model name or load path should be specified.")

    if not hasattr(args, "model_dir"):
        args.model_dir = os.path.join(args.log_dir, args.model_name)

    for path in [args.log_dir, args.model_dir]:
        if not os.path.exists(path):
            os.makedirs(path)


def remove_file(path: str) -> None:
    if os.path.exists(path):
        logger.info(f"[*] Remove: {path}")
        os.remove(path)


def save_args(args):
    param_path = os.path.join(args.model_dir, "params.json")

    logger.info(f"[*] MODEL dir: {args.model_dir}")
    logger.info(f"[*] PARAM path: {param_path}")

    with open(param_path, "w") as fp:
        json.dump(args.__dict__, fp, indent=4, sort_keys=True)


class TensorBoard(object):
    def __init__(self, model_dir: str) -> None:
        self.summary_writer = tb.FileWriter(model_dir)

    def scalar_summary(self, tag, value, step):
        summary = Summary(value=[Summary.Value(tag=tag, simple_value=value)])
        self.summary_writer.add_summary(summary, global_step=step)



