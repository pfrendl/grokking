import inspect
import math
import time
from itertools import product
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from enlighten import Manager
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        gain = 1 / math.sqrt(in_features)
        self.weight = nn.Parameter(gain * torch.randn((out_features, in_features)))
        self.bias = nn.Parameter(torch.zeros((out_features,)))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(input=x, weight=self.weight, bias=self.bias)


class ReLU(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        gain = math.sqrt(2)
        return gain * x.relu()


class Model(nn.Module):
    def __init__(self, num_tokens: int, hidden_size: int) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            Linear(in_features=2 * num_tokens, out_features=hidden_size),
            ReLU(),
            Linear(in_features=hidden_size, out_features=hidden_size),
            ReLU(),
            Linear(in_features=hidden_size, out_features=num_tokens),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.seq(x)


class AlgorithmicDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, inputs: Tensor, targets: Tensor, num_tokens: int) -> None:
        super().__init__()
        self.inputs = torch.cat(
            [
                F.one_hot(inputs[:, 0], num_classes=num_tokens),
                F.one_hot(inputs[:, 1], num_classes=num_tokens),
            ],
            dim=1,
        ).to(torch.float32)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.inputs[index], self.targets[index]


def create_data(
    algorithm: Callable[[int, int], int], num_tokens: int
) -> tuple[Tensor, Tensor]:
    inputs = []
    targets = []
    with np.errstate(all="ignore"):
        for x, y in product(np.arange(num_tokens), np.arange(num_tokens)):
            target = algorithm(x, y) % num_tokens
            if np.isfinite(target):
                inputs.append([x, y])
                targets.append(algorithm(x, y) % num_tokens)
    inputs_ten = torch.tensor(inputs, dtype=torch.int64).view((-1, 2))
    targets_ten = torch.tensor(targets, dtype=torch.int64)
    return inputs_ten, targets_ten


def main() -> None:
    device = torch.device("cuda")
    num_tokens = 97
    batch_size = 32
    hiddens_sizes = [64, 128, 256, 512, 1024]
    num_epochs = 100
    test_interval = 1
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    algos = [
        lambda x, y: x + y,
        lambda x, y: x - y,
        lambda x, y: x // y,
        lambda x, y: x // y if y % 2 == 1 else x - y,
        lambda x, y: x**2 + y**2,
        lambda x, y: x**2 + x * y + y**2,
        lambda x, y: x**2 + x * y + y**2 + x,
        lambda x, y: x**3 + x * y,
        lambda x, y: x**3 + x * y**2 + y,
    ]

    manager = Manager()

    test_pbar = manager.counter(
        total=len(algos), desc="Test", unit="algorithms", leave=False
    )
    for algo_idx, algorithm in enumerate(algos):
        fig = plt.figure(figsize=(10, 5), constrained_layout=True)
        axes = fig.subplot_mosaic(
            """
            AB
            """
        )

        inputs, targets = create_data(algorithm=algorithm, num_tokens=num_tokens)
        train_set_mask = torch.rand(targets.shape) < 0.5
        test_set_mask = ~train_set_mask

        train_set = AlgorithmicDataset(
            inputs=inputs[train_set_mask],
            targets=targets[train_set_mask],
            num_tokens=num_tokens,
        )
        test_set = AlgorithmicDataset(
            inputs=inputs[test_set_mask],
            targets=targets[test_set_mask],
            num_tokens=num_tokens,
        )

        train_loader = DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            persistent_workers=True,
        )
        test_loader = DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            persistent_workers=True,
        )

        experiment_pbar = manager.counter(
            total=len(hiddens_sizes), desc="Experiment", unit="models", leave=False
        )
        for hidden_size in hiddens_sizes:
            model = Model(num_tokens=num_tokens, hidden_size=hidden_size).to(device)
            optimizer = torch.optim.RAdam(
                params=model.parameters(), lr=0.001, weight_decay=0.001
            )

            training_pbar = manager.counter(
                total=num_epochs, desc="Training", unit="sec", leave=False
            )
            avg_train_losses = []
            avg_test_losses = []
            start = time.perf_counter()
            for epoch in range(num_epochs):
                train_epoch_pbar = manager.counter(
                    total=len(train_loader),
                    desc="Train epoch",
                    unit="iters",
                    leave=False,
                )
                train_losses = []
                for input, target in train_loader:
                    input, target = input.to(device), target.to(device)
                    output = model(input)
                    loss = -output.log_softmax(dim=1)[
                        range(output.shape[0]), target
                    ].mean(dim=0)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_losses.append(loss.item())
                    train_epoch_pbar.update()
                train_epoch_pbar.close()

                if epoch % test_interval == 0:
                    test_epoch_pbar = manager.counter(
                        total=len(test_loader),
                        desc="Test epoch",
                        unit="iters",
                        leave=False,
                    )
                    test_losses = []
                    with torch.no_grad():
                        for input, target in test_loader:
                            input, target = input.to(device), target.to(device)

                            output = model(input)
                            loss = -output.log_softmax(dim=1)[
                                range(output.shape[0]), target
                            ].mean(dim=0)
                            test_losses.append(loss.item())
                            test_epoch_pbar.update()
                    test_epoch_pbar.close()

                    avg_train_losses.append(
                        [epoch, sum(train_losses) / len(train_losses)]
                    )
                    avg_test_losses.append([epoch, sum(test_losses) / len(test_losses)])

                training_pbar.update(incr=1)

                avg_train_losses_np = torch.tensor(avg_train_losses).cpu().numpy()
                avg_test_losses_np = torch.tensor(avg_test_losses).cpu().numpy()
            training_pbar.close()

            axes["A"].plot(
                avg_train_losses_np[:, 0],
                avg_train_losses_np[:, 1],
                label=f"hidden_size = {hidden_size}",
            )
            axes["B"].plot(
                avg_test_losses_np[:, 0],
                avg_test_losses_np[:, 1],
                label=f"hidden_size = {hidden_size}",
            )
            experiment_pbar.update()
        experiment_pbar.close()

        fig.suptitle(inspect.getsource(algorithm).split(":")[1][:-2])

        axes["A"].set_title(f"Train set")
        axes["A"].set_xlabel("Epochs")
        axes["A"].set_ylabel("Cross-entropy loss")
        axes["A"].grid(True)
        axes["A"].legend(loc="upper right")

        axes["B"].set_title(f"Test set")
        axes["B"].set_xlabel("Epochs")
        axes["B"].set_ylabel("Cross-entropy loss")
        axes["B"].grid(True)
        axes["B"].legend(loc="upper right")

        plt.savefig(out_dir / f"algorithm{algo_idx}.png")
        plt.close(fig)
        test_pbar.update()
    test_pbar.close()


if __name__ == "__main__":
    main()
