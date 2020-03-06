import logging
import torch
import os
import glob
from argparse import Namespace
from typing import Any, Tuple
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F

from model import ConvolutionNet, FullConnectedNet, CoOccurrenceNet
from utils.utils import TensorBoard, remove_file

logger = logging.getLogger(__name__)
Int_Tuple = Tuple[int, int]
List_Tuple = Tuple[list, list]


def get_optimizer(name: str) -> torch.optim.Optimizer:
    if name.lower() == "sgd":
        optim = torch.optim.SGD
    elif name.lower() == "adam":
        optim = torch.optim.Adam
    elif name.lower() == "rmsprop":
        optim = torch.optim.RMSprop
    else:
        raise NotImplementedError
    return optim


def calc_accurecy(pred: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    pred = torch.argmax(pred, dim=1)
    accurecy = torch.sum(pred == label).float() / pred.shape[0]
    return accurecy


class Trainer(object):
    def __init__(self, args: Namespace, dataset: dict) -> None:
        self.args = args
        self.dataset = dataset
        self.cuda = self.args.cuda
        self.train_data = dataset["train"]
        self.test_data = dataset["test"]
        self.val_data = dataset["test"]

        if self.args.use_tensorboard and self.args.mode == "train":
            self.tb = TensorBoard(self.args.model_dir)
        else:
            self.tb = None

        self.build_model()

        if self.args.load_path:
            self.load_model()

    def build_model(self) -> None:
        self.start_epoch = self.epoch = 0
        self.step = 0
        if self.args.model_type == "FC":
            self.model = FullConnectedNet()
        elif self.args.model_type == "Conv":
            self.model = ConvolutionNet()
        elif self.args.model_type == "Conn":
            self.model = CoOccurrenceNet()
        else:
            raise NotImplementedError

        if self.args.cuda:
            self.model.cuda()

        logger.info(f"[*] Number of Parameters: {self.count_parameters}")

    def train(self) -> None:
        optimizer = get_optimizer(self.args.optim)
        self.optim = optimizer(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr
        )

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            self.train_model()
            if self.epoch % self.args.save_epoch == 0:
                score = self.test("val")
                self.save_model(score)

    def train_model(self) -> None:
        total_loss = 0
        model = self.model
        model.train()

        pbar = tqdm(total=len(self.train_data), desc="train_model")

        for step, data_batch in enumerate(self.train_data):
            image, label = data_batch[0], data_batch[1]
            image, label = image.float(), label.long()
            if self.args.cuda:
                image, label = image.cuda(), label.cuda()

            pred = self.model(image)
            loss = F.cross_entropy(input=pred, target=label)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            pbar.set_description(f"train_model| loss: {loss.data: 5.3f}")

            if step % self.args.log_step == 0:
                logger.info(f"| epoch {self.epoch:3d} | lr {self.args.lr:8.6f} "
                            f"| loss {loss.data:.2f}")

                if self.tb is not None:
                    self.tb.scalar_summary("model/loss", loss.data, self.step)

            self.step += 1
            pbar.update(1)

    def test(self, mode: str) -> float:
        self.model.eval()
        pbar = tqdm(total=len(self.test_data), desc="test_model")
        score = 0

        if mode == "val":
            data = self.val_data
        elif mode == "test":
            data = self.test_data
        else:
            raise NotImplementedError

        for idx, data_batch in enumerate(data):
            image, label = data_batch[0], data_batch[1]
            image, label = image.float(), label.long()
            if self.args.cuda:
                image, label = image.cuda(), label.cuda()

            with torch.no_grad():
                pred = self.model(image)
                accurecy = calc_accurecy(pred, label)
                score += accurecy

            pbar.update(1)

        score /= len(data)
        if mode == "val" and self.tb is not None:
            self.tb.scalar_summary(f"test/{mode}", score, self.epoch)
        return score

    def save_model(self, save_criteria_score: float = None) -> None:
        torch.save(self.model.state_dict(), self.model_save_path)
        logger.info(f"[*] SAVED: {self.model_save_path}")
        epochs, steps = self.get_save_models_info()

        if save_criteria_score is not None:
            if os.path.exists(self.checkpoint_path):
                checkpoint_tracker = torch.load(self.checkpoint_path)
            else:
                checkpoint_tracker = {}
            key = f"{self.epoch}_{self.step}"
            value = save_criteria_score
            checkpoint_tracker[key] = value
            if len(epochs) > self.args.max_save_num:
                low_value = 10000.0
                remove_key = None
                for key, value in checkpoint_tracker.items():
                    if low_value > value:
                        remove_key = key
                        low_value = value

                del checkpoint_tracker[remove_key]

                remove_epoch = remove_key.split("_")[0]
                remove_step = remove_key.split("_")[1]
                path = glob.glob(os.path.join(self.args.model_dir, f"*_epoch{remove_epoch}_step{remove_step}.pth"))
                for p in path:
                    remove_file(p)

            torch.save(checkpoint_tracker, self.checkpoint_path)
        else:
            for epoch in epochs[:-self.args.max_save_num]:
                paths = glob.glob(os.path.join(self.args.model_dir, f"*_epoch{epoch}_*.pth"))
                for path in paths:
                    remove_file(path)

    def load_model(self) -> None:
        if self.args.cuda:
            map_location = lambda storage, loc: storage
        else:
            map_location = None

        if self.args.load_path.endswith(".pth"):
            self.model.load_state_dict(
                torch.load(self.args.load_path, map_location=map_location)
            )
            self.epoch, self.step = self._get_save_model_info(self.args.load_path)
            self.start_epoch = self.epoch
            logger.info(f"[*] LOADED: {self.args.load_path}")
        else:
            if os.path.exists(self.checkpoint_path):
                checkpoint_tracker = torch.load(self.checkpoint_path)
                best_key = None
                best_score = -10000.0
                for key, value in checkpoint_tracker.items():
                    if best_score < value:
                        best_key = key
                        best_score = value

                self.epoch = self.start_epoch = int(best_key.split("_")[0])
                self.step = int(best_key.split("_")[1])
            else:
                epochs, steps = self.get_save_models_info()

                if len(epochs) == 0:
                    logger.warning(f"[!] No checkpoint found in {self.args.model_dir}")
                    return

                self.epoch = self.start_epoch = max(epochs)
                self.step = max(steps)

            self.model.load_state_dict(
                torch.load(self.load_path, map_location=map_location)
            )
            logger.info(f"[*] LOADED: {self.load_path}")

    def get_save_models_info(self) -> List_Tuple:
        paths = glob.glob(os.path.join(self.args.model_dir, "*.pth"))
        paths.sort()

        epochs = []
        steps = []
        for path in paths:
            epoch, step = self._get_save_model_info(path)
            epochs.append(epoch)
            steps.append(step)

        epochs.sort()
        steps.sort()
        return epochs, steps

    @staticmethod
    def _get_save_model_info(path: str) -> Int_Tuple:

        def get_number(item, delimiter, idx, replace_word, must_contain=''):
            if must_contain in item:
                return int(item.split(delimiter)[idx].replace(replace_word, ''))

        basename = os.path.basename(path.rsplit('.', 1)[0])
        epoch = get_number(basename, "_", 1, "epoch")
        step = get_number(basename, "_", 2, "step", "model")

        return epoch, step

    @property
    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    @property
    def checkpoint_path(self):
        return os.path.join(self.args.model_dir, "checkpoint_tracker.dat")

    @property
    def load_path(self):
        return f"{self.args.load_path}/model_epoch{self.epoch}_step{self.step}.pth"

    @property
    def model_save_path(self):
        return f"{self.args.model_dir}/model_epoch{self.epoch}_step{self.step}.pth"

