import argparse
import torch
import torchvision
import wandb

from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

from model import resnet18
from optimizer.adam import  ApproxAdam
from optimizer.sgd import ApproxSGD
from optimizer.rank_ef import RankEF
from optimizer.ef21 import EF21, EF21Plus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="sgd")
    parser.add_argument("--approx", choices=["none", "power_sgd", "power_ef", "power_ef_plus"], default="none")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epoches", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--cifar_root", type=str, default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--optim_rank", type=int, default=4)
    return parser.parse_args()


def weight_decay(model):
    weight_norm = 0
    for n, p in model.named_parameters():
        if "bn" not in n:
            weight_norm = weight_norm + p.norm() ** 2
    return weight_norm


def main(args):
    wandb.init(project='power_ef', entity='artek-chumak', config=args, mode="online")

    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    valid_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    train_data = CIFAR10(args.cifar_root, download=True, transform=train_transform)
    train_dataloader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        sampler=torch.utils.data.RandomSampler(train_data),
        num_workers=4
    )
    valid_data = CIFAR10(args.cifar_root, train=False, download=True, transform=valid_transform)
    valid_dataloader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )

    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    model = resnet18()
    if torch.cuda.is_available():
        model = model.to("cuda")
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

    def validate():
        with torch.no_grad():
            accurate = 0
            for batch, target in tqdm(valid_dataloader):
                if torch.cuda.is_available():
                    batch = batch.to("cuda")
                pred = model(batch)
                if torch.cuda.is_available():
                    pred = pred.to("cpu")
                pred = torch.argmax(pred, -1)
                accurate += (pred == target).sum().item()
            accurate = accurate / len(valid_data)
            wandb.log({"Valid Acc": accurate})
        print(f"Accuracy: {accurate:.4f}")

    validate()
    for e in range(args.num_epoches):
        train_trange = tqdm(train_dataloader, total=len(train_dataloader), desc="Epoch: 0; Loss: inf")
        for batch, target in train_trange:
            optimizer.zero_grad()
            if torch.cuda.is_available():
                batch = batch.to("cuda")
                target = target.to("cuda")
            pred = model(batch)
            loss = loss_fn(pred, target)
            if args.weight_decay > 0:
                loss = loss + weight_decay(model) * args.weight_decay
            loss.backward()
            if torch.cuda.is_available():
                loss = loss.to("cpu")
            optimizer.step()
            train_trange.set_description(f"Epoch: {e}; Loss: {loss.item():.4f}")
            wandb.log({"Train Loss": loss})
        validate()


if __name__ == "__main__":
    args = parse_args()
    main(args)
