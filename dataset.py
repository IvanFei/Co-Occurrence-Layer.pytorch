
import random
import torch
import numpy as np
from torch.utils import data
from typing import Tuple
import matplotlib.pyplot as plt

Data_List = Tuple[np.ndarray, int]
Loader_List = Tuple[data.Dataset, data.DataLoader]


class ToyData(data.Dataset):

    IMAGE_SIZE = 10
    NUM_BINS = 4
    HIGH_DIST = 0.4
    LOW_DIST = 0.1
    NUMBER_TRAIN_IMAGES = 6000
    NUMBER_TEST_IMAGES = 1000

    def __init__(self, mode: str = "train") -> None:
        super(ToyData, self).__init__()
        self.mode = mode
        if self.mode == "train":
            self.num_data = self.NUMBER_TRAIN_IMAGES
        elif self.mode == "test":
            self.num_data = self.NUMBER_TEST_IMAGES
        else:
            raise NotImplementedError("This mode: {} has't been implemented".format(self.mode))

    def __getitem__(self, item: int) -> Data_List:
        rand_seed = random.uniform(0, 1)
        label = 0 if rand_seed <= 0.5 else 1
        hist = self._init_histogram(label)
        image = self._create_image(hist)
        image = np.random.permutation(image)
        image = self._normalize(image)

        return image, label

    def __len__(self) -> int:
        return self.num_data

    def _init_histogram(self, label: int) -> list:
        if label == 0:
            hist = [self.HIGH_DIST, self.LOW_DIST, self.LOW_DIST, self.HIGH_DIST]
        else:
            hist = [self.LOW_DIST, self.HIGH_DIST, self.HIGH_DIST, self.LOW_DIST]

        assert (sum(hist) == 1)

        return hist

    def _create_image(self, hist: list) -> np.ndarray:
        num_pixel = self.IMAGE_SIZE * self.IMAGE_SIZE
        hist_count = np.floor(num_pixel * np.array(hist))

        list_values = []
        for i in range(self.NUM_BINS):
            ones_vec = np.ones(int(hist_count[i]))
            single_value_vec = np.array(i * ones_vec, dtype=np.float)
            list_values = np.append(list_values, single_value_vec)

        return list_values

    @staticmethod
    def _normalize(image: np.ndarray) -> np.ndarray:
        ret = (image - image.min()) / (image.max() - image.min())
        return ret


def get_loader(batch_size: int, mode: str = "train", num_threads: int = 4) -> Loader_List:
    if mode == "train":
        shuffle = True
    else:
        shuffle = False

    dataset = ToyData(mode=mode)
    data_loader = data.DataLoader(dataset, batch_size=batch_size,
                                  shuffle=shuffle, num_workers=num_threads)
    return dataset, data_loader


if __name__ == "__main__":
    dataset, data_loader = get_loader(batch_size=32, mode="test")
    for batch in data_loader:
        print(f"[*] image shape: {batch[0].shape}")
        print(f"[*] label is: {batch[1]}")
        img = batch[0]
        break

    print(f"[*] image sample: {img[0]}")
    plt.imshow(img[0].reshape([ToyData.IMAGE_SIZE, ToyData.IMAGE_SIZE]))
    plt.show()


