import argparse
import torch
import torchvision
import wandb

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from model import SmallCNN
from optimizer.sgd import ApproxSGD
from optimizer.rank_ef import RankEF
from optimizer.ef21 import EF21


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["sgd", "power_sgd", "power_ef"], default="sgd")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epoches", type=int, default=3)
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--mnist_root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optim_rank", type=int, default=4)
    return parser.parse_args()


def main(args):
    wandb.init(project='power_ef', entity='artek-chumak', config=args)

    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model = SmallCNN()
    wandb.watch(model, log_freq=100)
    optimizer = ApproxSGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    if args.optimizer == "sgd":
        ef = None
    elif args.optimizer == "power_sgd":
        ef = RankEF(rank=args.optim_rank)
    elif args.optimizer == "power_ef":
        ef = EF21(rank=args.optim_rank)
    else:
        raise NotImplementedError()
    if ef is not None:
        ef.add_groups(optimizer)

    loss_fn = torch.nn.CrossEntropyLoss()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    train_data = MNIST(args.mnist_root, download=True, transform=transform)
    train_dataloader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    valid_data = MNIST(args.mnist_root, train=False, download=True, transform=transform)
    valid_dataloader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=True
    )

    def validate():
        with torch.no_grad():
            accurate = 0
            for batch, target in tqdm(valid_dataloader):
                pred = model(batch)
                pred = torch.argmax(pred, -1)
                accurate += (pred == target).sum().item()
            accurate = accurate / len(valid_data)
            wandb.log({"Valid Acc": accurate})
        print(f"Accuracy: {accurate:.4f}")

    validate()
    for e in range(args.num_epoches):
        train_trange = tqdm(train_dataloader, total=len(train_dataloader))
        for batch, target in train_trange:
            pred = model(batch)
            loss = loss_fn(pred, target)
            wandb.log({"Train Loss": loss})
            loss.backward()
            optimizer.step()
            train_trange.set_description(f"Epoch: {e}; Loss: {loss.item():.4f}")
            optimizer.zero_grad()
        validate()


if __name__ == "__main__":
    args = parse_args()
    main(args)
