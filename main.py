import os
import torch
import config
from dataset import get_loader
from trainer import Trainer
from utils.utils import prepare_dirs, save_args


def main(args):
    prepare_dirs(args)
    torch.random.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    data = {}
    _, data["train"] = get_loader(args.batch_size, "train", args.num_threads)
    _, data["test"] = get_loader(args.batch_size, "test", args.num_threads)

    trainer = Trainer(args, data)

    if args.mode == "train":
        save_args(args)
        trainer.train()
    elif args.mode == "test":
        if not args.load_path:
            raise Exception(f"[!] You should specify `load_path` to load a pretrained model.")
        else:
            trainer.test(args.mode)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args, unparsed = config.get_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    main(args)

