from data.dataset import MyDataset
from torch.utils import data
import torchvision.transforms as transforms


def get_data_loader(datadir, args, batch_size=32, dataidxs=None):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    train_data = MyDataset(
        args.traindir,
        datadir + f"{args.dataset}_train.csv",
        args.dataset,
        transform,
        dataidxs,
    )
    test_data = MyDataset(
        args.testdir, datadir + f"{args.dataset}_test.csv", args.dataset, transform
    )

    train_loader = data.DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )
    test_loader = data.DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
    )

    return train_loader, test_loader
