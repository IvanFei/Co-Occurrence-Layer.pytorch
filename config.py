import argparse
import logging

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
net = parser.add_argument_group("Net")
net.add_argument("--model_type", type=str, default="Conn", choices=["FC", "Conv", "Conn"])

data = parser.add_argument_group("Data")
data.add_argument("--num_threads", type=int, default=4)

learn = parser.add_argument_group("Learning")
learn.add_argument("--mode", type=str, default="train", choices=["train", "test"])
learn.add_argument("--batch_size", type=int, default=32)
learn.add_argument("--max_epoch", type=int, default=50)
learn.add_argument("--optim", type=str, default="sgd")
learn.add_argument("--lr", type=float, default=0.001)

misc = parser.add_argument_group("Misc")
misc.add_argument("--model_name", type=str, default="")
misc.add_argument("--load_path", type=str, default="")
misc.add_argument("--log_dir", type=str, default="logs")
misc.add_argument("--random_seed", type=int, default=0)
misc.add_argument("--log_step", type=int, default=10)
misc.add_argument("--save_epoch", type=int, default=10)
misc.add_argument("--max_save_num", type=int, default=5)

gpu = parser.add_argument_group("GPU")
gpu.add_argument("--gpu_id", type=int, default=0)

vis = parser.add_argument_group("Visulization")
vis.add_argument("--use_tensorboard", type=bool, default=True)


def get_args():
    args, unparsed = parser.parse_known_args()
    if args.gpu_id < 0:
        setattr(args, "cuda", False)
    else:
        setattr(args, "cuda", True)

    if len(unparsed) > 1:
        logger.info(f"[*] Unparsed args: {unparsed}")

    return args, unparsed
