import argparse
import torch
import torchvision
import wandb

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from model import SmallCNN
from optimizer.adam import ApproxAdam
from optimizer.sgd import ApproxSGD
from optimizer.rank_ef import RankEF
from optimizer.ef21 import EF21, EF21Plus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--approx", choices=["none", "power_sgd", "power_ef", "power_ef_plus"], default="none")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epoches", type=int, default=3)
    parser.add_argument("--momentum", type=float, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mnist_root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optim_rank", type=int, default=4)
    return parser.parse_args()


def main(args):
    wandb.init(project='power_ef', entity='artek-chumak', config=args, mode="online")

    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model = SmallCNN()
    wandb.watch(model, log_freq=100)
    if args.optimizer == "adam":
        optimizer = ApproxAdam(model.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = ApproxSGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    else:
        raise NotImplementedError()

    if args.approx == "none":
        ef = None
    elif args.approx == "power_sgd":
        ef = RankEF(rank=args.optim_rank)
    elif args.approx == "power_ef":
        ef = EF21(rank=args.optim_rank)
    elif args.approx == "power_ef_plus":
        ef = EF21Plus(rank=args.optim_rank)
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
