#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Binary MNIST trial for a stochastic PRNN classifier.

The script trains two models on the same small 0-vs-1 MNIST subset:

* ``PRNNClassifier``: encodes each image to GP length scales, samples a strain
  path on every forward pass, runs the path through the J2 material layer, and
  decodes equivalent plastic strains to a binary class probability.
* ``MLPClassifier``: a direct fully-connected image classifier used as a simple
  baseline.

Example
-------
python mnist-prnn-classifier-demo.py --train-size 128 --test-size 64 --epochs 5
"""

import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from prnn import PRNNClassifier


class MLPClassifier(torch.nn.Module):
    """Small direct MNIST baseline using the same flattened inputs."""

    def __init__(
        self,
        image_shape,
        hidden_sizes=(64, 32),
        device=torch.device('cpu'),
    ):
        super().__init__()
        if isinstance(image_shape, int):
            image_shape = (image_shape,)
        image_size = int(np.prod(image_shape))
        layers = [torch.nn.Flatten()]
        in_features = image_size
        for hidden_size in hidden_sizes:
            layers.extend([
                torch.nn.Linear(in_features, hidden_size, device=device),
                torch.nn.ReLU(),
            ])
            in_features = hidden_size
        layers.extend([
            torch.nn.Linear(in_features, 1, device=device),
            torch.nn.Sigmoid(),
        ])
        self.net = torch.nn.Sequential(*layers)

    def forward(self, images):
        return self.net(images)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--digits',
        nargs=2,
        type=int,
        default=(0, 1),
        help='two MNIST digits to classify',
    )
    parser.add_argument(
        '--train-size',
        type=int,
        default=128,
        help='number of training samples',
    )
    parser.add_argument('--test-size', type=int, default=64, help='number of test samples')
    parser.add_argument('--batch-size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=5, help='training epochs for each model')
    parser.add_argument('--image-size', type=int, default=14, help='downsampled square image size')
    parser.add_argument('--seq-len', type=int, default=24, help='sampled GP strain path length')
    parser.add_argument(
        '--mat-pts',
        type=int,
        default=6,
        help='number of material points in the PRNN',
    )
    parser.add_argument(
        '--mc-samples',
        type=int,
        default=8,
        help='PRNN Monte Carlo samples during evaluation',
    )
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--data-dir', default='data', help='MNIST download/cache directory')
    parser.add_argument(
        '--figure',
        default='mnist-prnn-confusion.png',
        help='confusion-matrix output path',
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_binary_mnist(data_dir, digits, train_size, test_size, image_size, dtype):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    train_tensors = _select_digits(train_data, digits, train_size, dtype)
    test_tensors = _select_digits(test_data, digits, test_size, dtype)
    return TensorDataset(*train_tensors), TensorDataset(*test_tensors)


def _select_digits(dataset, digits, size, dtype):
    images = []
    labels = []
    digit_to_label = {digits[0]: 0.0, digits[1]: 1.0}
    for image, target in dataset:
        target = int(target)
        if target in digit_to_label:
            images.append(image.to(dtype=dtype))
            labels.append([digit_to_label[target]])
        if len(images) == size:
            break
    return torch.stack(images), torch.tensor(labels, dtype=dtype)


def train_classifier(model, loader, device, epochs, lr, name):
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'{name} epoch {epoch + 1:03d}: loss={running_loss / len(loader):.4f}')


def predict_probabilities(model, loader, device, mc_samples=1):
    model.eval()
    probabilities = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if mc_samples == 1:
                batch_probabilities = model(images)
            else:
                draws = [model(images) for _ in range(mc_samples)]
                batch_probabilities = torch.stack(draws).mean(dim=0)
            probabilities.append(batch_probabilities.cpu())
            targets.append(labels.cpu())
    return torch.cat(probabilities), torch.cat(targets)


def confusion_matrix(probabilities, targets):
    predictions = (probabilities >= 0.5).to(torch.int64).view(-1)
    targets = targets.to(torch.int64).view(-1)
    matrix = torch.zeros((2, 2), dtype=torch.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def plot_confusion_matrices(matrices, titles, digits, filename):
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4))
    if len(matrices) == 1:
        axes = [axes]
    for ax, matrix, title in zip(axes, matrices, titles):
        matrix_np = matrix.numpy()
        image = ax.imshow(matrix_np, cmap='Blues')
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks([0, 1], labels=[str(digits[0]), str(digits[1])])
        ax.set_yticks([0, 1], labels=[str(digits[0]), str(digits[1])])
        for row in range(2):
            for col in range(2):
                ax.text(
                    col,
                    row,
                    str(matrix_np[row, col]),
                    ha='center',
                    va='center',
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f'Saved confusion matrices to {filename}')


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.get_default_dtype()

    train_dataset, test_dataset = load_binary_mnist(
        args.data_dir,
        tuple(args.digits),
        args.train_size,
        args.test_size,
        args.image_size,
        dtype,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    image_shape = train_dataset.tensors[0].shape[1:]

    prnn = PRNNClassifier(
        image_shape=image_shape,
        n_matpts=args.mat_pts,
        seq_len=args.seq_len,
        device=device,
    ).to(device)
    mlp = MLPClassifier(image_shape=image_shape, device=device).to(device)

    train_classifier(prnn, train_loader, device, args.epochs, args.lr, 'PRNN')
    train_classifier(mlp, train_loader, device, args.epochs, args.lr, 'MLP')

    prnn_probabilities, targets = predict_probabilities(
        prnn,
        test_loader,
        device,
        args.mc_samples,
    )
    mlp_probabilities, _ = predict_probabilities(mlp, test_loader, device, 1)
    prnn_matrix = confusion_matrix(prnn_probabilities, targets)
    mlp_matrix = confusion_matrix(mlp_probabilities, targets)

    print('PRNN confusion matrix:\n', prnn_matrix.numpy())
    print('MLP confusion matrix:\n', mlp_matrix.numpy())
    plot_confusion_matrices(
        [prnn_matrix, mlp_matrix],
        [f'PRNN ({args.mc_samples} MC samples)', 'MLP baseline'],
        tuple(args.digits),
        args.figure,
    )


if __name__ == '__main__':
    main()
