import argparse
import copy
import datetime
import json
import os
import random

import numpy as np
import torch

from models.model import init_nets
from data.loader import get_data_loader
from utils.utils import mkdirs, partition_data
from utils.trainer import fit, test

import warnings

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="tumor", help="dataset used for training"
    )
    parser.add_argument(
        "--partition", type=str, default="noniid", help="the data partitioning strategy"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="total sum of input batch size for training (default: 128)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="number of local epochs")
    parser.add_argument(
        "--n_parties",
        type=int,
        default=5,
        help="number of workers in a distributed cluster",
    )
    parser.add_argument(
        "--comm_round", type=int, default=5, help="number of maximum communication roun"
    )
    parser.add_argument("--init_seed", type=int, default=0, help="Random seed")
    parser.add_argument("--datadir", type=str, default="./data/", help="csv files")
    parser.add_argument(
        "--traindir",
        type=str,
        default="../../Datasets/Brain_Tumor/Training",
        help="training file directory",
    )
    parser.add_argument(
        "--testdir",
        type=str,
        default="../../Datasets/Brain_Tumor/Testing",
        help="testing file directory",
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
        "--beta",
        type=float,
        default=0.6,
        help="The parameter for the dirichlet distribution for data partitioning",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="The device to run the program"
    )
    parser.add_argument("--optimizer", type=str, default="adam", help="the optimizer")
    parser.add_argument(
        "--out_dim",
        type=int,
        default=4,
        help="the output dimension for the projection layer",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Size of hidden unit"
    )
    parser.add_argument(
        "--sample_fraction",
        type=float,
        default=1.0,
        help="how many clients are sampled in each round",
    )
    args = parser.parse_args()
    return args


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)


def local_train_net(
    nets,
    args,
    net_dataidx_map,
    train_dl_local_dict,
):
    for net_id, net in nets.items():
        dataidxs = net_dataidx_map[net_id]
        print("Training network %s. n_training: %d" % (str(net_id), len(dataidxs)))
        train_dl_local = train_dl_local_dict[net_id]
        fit(net, train_dl_local, args.epochs, args.lr, args.optimizer, args.device)

    return nets


if __name__ == "__main__":
    args = get_args()

    # Tumor
    classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
    # Covid
    # classes = ["Covid", "Lung Opacity", "Normal", "Viral Pneumonia"]

    # Create directory to save log and model
    mkdirs(args.logdir)
    argument_path = (
        f"{args.dataset}-{args.n_parties}_arguments-%s.json"
        % datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")
    )

    # Save arguments
    with open(os.path.join(args.logdir, argument_path), "w") as f:
        json.dump(str(args), f)

    # Set seed
    set_seed(args.init_seed)

    # Data partitioning with respect to the number of parties
    print("Partitioning data")
    (X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts) = (
        partition_data(
            args.datadir, args.partition, args.n_parties, args, beta=args.beta
        )
    )

    n_party_per_round = int(args.n_parties * args.sample_fraction)
    party_list = [i for i in range(args.n_parties)]
    party_list_rounds = []

    if n_party_per_round != args.n_parties:
        for i in range(args.comm_round):
            party_list_rounds.append(random.sample(party_list, n_party_per_round))
    else:
        for i in range(args.comm_round):
            party_list_rounds.append(party_list)

    n_classes = len(np.unique(y_train))

    # Get global dataloader (only used for evaluation)
    (train_dl_global, test_dl) = get_data_loader(args.datadir, args, args.batch_size)

    print("len train_dl_global:", len(X_train))

    # Initializing net from each local party.
    print("Initializing nets")
    nets = init_nets(args.n_parties, args)
    global_models = init_nets(1, args)

    global_model = global_models[0]
    n_comm_rounds = args.comm_round

    train_dl_local_dict = {}
    net_id = 0

    # Distribute dataset and dataloader to each local party
    for net in nets:
        dataidxs = net_dataidx_map[net_id]
        train_dl_local, _ = get_data_loader(
            args.datadir, args, args.batch_size, dataidxs
        )
        train_dl_local_dict[net_id] = train_dl_local
        net_id += 1

    # Main training communication loop.
    for round in range(n_comm_rounds):
        print("in comm round:" + str(round))
        party_list_this_round = party_list_rounds[round]

        # Download global model from (virtual) central server
        global_w = global_model.state_dict()
        nets_this_round = {k: nets[k] for k in party_list_this_round}
        for net in nets_this_round.values():
            net.load_state_dict(global_w)

        # Train local model with local data
        local_train_net(
            nets_this_round,
            args,
            net_dataidx_map,
            train_dl_local_dict,
        )

        total_data_points = sum(
            [len(net_dataidx_map[r]) for r in range(args.n_parties)]
        )
        fed_avg_freqs = [
            len(net_dataidx_map[r]) / total_data_points for r in range(args.n_parties)
        ]

        # Averaging the local models' parameters to get global model
        for net_id, net in enumerate(nets_this_round.values()):
            net_para = net.state_dict()
            if net_id == 0:
                for key in net_para:
                    global_w[key] = net_para[key] * fed_avg_freqs[net_id]
            else:
                for key in net_para:
                    global_w[key] += net_para[key] * fed_avg_freqs[net_id]

        global_model.load_state_dict(copy.deepcopy(global_w))

        # Evaluating the global model
        test(
            global_model,
            test_dl,
            classes,
            plot_path=f"{args.dataset}_{args.partition}_{args.beta}",
            verbose=False,
            device=args.device,
        )

    test(
        global_model,
        test_dl,
        classes,
        plot_path=f"{args.dataset}_{args.partition}_{args.beta}",
        device=args.device,
    )

    # Save the final round's local and global models
    torch.save(
        global_model.state_dict(),
        args.modeldir + "globalmodel_" + args.dataset + "_" + args.partition + ".pth",
    )
