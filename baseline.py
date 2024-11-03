import argparse
import datetime
import json
import os
import random
import numpy as np

import torch

from data.loader import get_data_loader
from utils.utils import mkdirs
from utils.trainer import fit, test
from models.model import init_nets

import warnings

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="tumor", help="dataset used for training"
    )
    parser.add_argument("--datadir", type=str, default="./data/", help="csv files")
    parser.add_argument(
        "--traindir", type=str, default="../../Datasets/Brain_Tumor/Training"
    )
    parser.add_argument(
        "--testdir", type=str, default="../../Datasets/Brain_Tumor/Testing"
    )
    parser.add_argument(
        "--logdir",
        type=str,
        required=False,
        default="./logs/",
        help="Log directory path",
    )
    parser.add_argument(
        "--modeldir",
        type=str,
        required=False,
        default="./models/",
        help="Model directory path",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size for training (default: 32)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--optimizer", type=str, default="adam", help="the optimizer")
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="hidden units in fc layer"
    )
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--out_dim", type=int, default=4, help="the output dimension for the fc layer"
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="The device to run the program"
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


if __name__ == "__main__":
    # args = get_covid_args()
    args = get_args()

    # Tumor
    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    # Covid
    # classes = ["Covid", "Lung Opacity", "Normal", "Viral Pneumonia"]

    # Create directory to save log and model
    mkdirs(args.logdir)
    argument_path = (
        f"{args.dataset}_arguments-%s.json"
        % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    )

    # Save arguments
    with open(os.path.join(args.logdir, argument_path), "w") as f:
        json.dump(str(args), f)

    set_seed(args.init_seed)

    print("-" * 50)
    print("- Loading data...")

    train_dl, test_dl = get_data_loader(args.datadir, args, batch_size=args.batch_size)

    # Initializing net
    print("Initializing nets")
    models = init_nets(1, args)

    model = models[0]

    fit(
        model,
        train_dl,
        args.epochs,
        args.lr,
        args.optimizer,
        device=args.device,
    )

    test(model, test_dl, classes, args, plot_path=args.dataset, device=args.device)

    # Save the final round's local and global models
    torch.save(model.state_dict(), args.modeldir + "basemodel_" + args.dataset + ".pth")
