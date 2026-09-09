#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MNIST trial for a stochastic multi-class PRNN classifier.

The script trains three models on the same configurable MNIST digit subset:

* A linear-elastic ``PRNNClassifier`` followed by a J2 plastic PRNN. Each
  encodes an image to a strain path and decodes the final material-point
  stresses.
* The plastic ``PRNNClassifier`` can instead decode internal variables when
  requested.
* ``PRNNClassifier`` in GP mode encodes each image to GP length scales, samples a strain
  path on every forward pass, runs the path through the J2 material layer, and
  decodes the selected material state features to class logits.
* ``MLPClassifier``: a direct fully-connected image classifier used as a simple
  baseline.

Examples
--------
python mnist-prnn-classifier-demo.py --digits 0 1 --train-size 128
python mnist-prnn-classifier-demo.py --digits 0 1 2 3 --train-size 512
python mnist-prnn-classifier-demo.py --sigma-f-mode image
python mnist-prnn-classifier-demo.py --image-parametrization timesignal

In addition to confusion matrices, the script saves a row-per-test-sample
diagnostic plot showing each image, its sampled PRNN strain paths, and the
corresponding PRNN prediction/target labels.
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
        n_classes,
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
        layers.append(torch.nn.Linear(in_features, n_classes, device=device))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, images, return_logits=True):
        return self.net(images)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--digits',
        nargs='+',
        type=int,
        default=(0, 1),
        help='MNIST digits to classify; pass any number from 2 to 10',
    )
    parser.add_argument(
        '--train-size',
        type=int,
        default=128,
        help='number of training samples across all selected digits',
    )
    parser.add_argument(
        '--test-size',
        type=int,
        default=64,
        help='number of test samples across all selected digits',
    )
    parser.add_argument('--batch-size', type=int, default=16, help='mini-batch size')
    parser.add_argument('--epochs', type=int, default=5, help='training epochs for each model')
    parser.add_argument('--image-size', type=int, default=14, help='downsampled square image size')
    parser.add_argument('--seq-len', type=int, default=24, help='sampled GP strain path length')
    parser.add_argument(
        '--image-parametrization',
        choices=('gp', 'timesignal'),
        default='gp',
        help='map images to GP-sampled or direct rasterized strain paths',
    )
    parser.add_argument(
        '--timesignal-smoothing-steps',
        type=int,
        default=0,
        help='linearly interpolated steps inserted between timesignal pixels',
    )
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
    parser.add_argument(
        '--decoder-features',
        choices=('epspeq', 'epsp', 'both', 'stress'),
        default='stress',
        help='features decoded by the plastic PRNN (linear always uses stress)',
    )
    parser.add_argument(
        '--sigma-f-mode',
        choices=('constant', 'image'),
        default='constant',
        help='use constant trainable or image-dependent GP amplitudes',
    )
    parser.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--data-dir', default='data', help='MNIST download/cache directory')
    parser.add_argument(
        '--figure',
        default='mnist-prnn-confusion.png',
        help='confusion-matrix output path',
    )
    parser.add_argument(
        '--sample-figure',
        default='mnist-prnn-test-samples.png',
        help='row-per-test-sample PRNN diagnostic figure output path',
    )
    parser.add_argument(
        '--loss-figure',
        default='mnist-prnn-test-loss.png',
        help='test-loss curve output path',
    )
    parser.add_argument(
        '--loss-file',
        default='mnist-prnn-test-loss.txt',
        help='tab-separated per-epoch test losses',
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def validate_digits(digits):
    digits = tuple(dict.fromkeys(digits))
    if len(digits) < 2:
        raise ValueError('Select at least two distinct digits with --digits.')
    invalid_digits = [digit for digit in digits if digit < 0 or digit > 9]
    if invalid_digits:
        raise ValueError(f'MNIST digits must be between 0 and 9: {invalid_digits}')
    return digits


def load_mnist_subset(data_dir, digits, train_size, test_size, image_size, dtype):
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
    digit_to_label = {digit: index for index, digit in enumerate(digits)}
    for image, target in dataset:
        target = int(target)
        if target in digit_to_label:
            images.append(image.to(dtype=dtype))
            labels.append(digit_to_label[target])
        if len(images) == size:
            break
    if len(images) < size:
        raise ValueError(
            f'Requested {size} samples for digits {digits}, but found {len(images)}.'
        )
    return torch.stack(images), torch.tensor(labels, dtype=torch.long)


def evaluate_loss(model, loader, device, mc_samples=1):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if mc_samples == 1:
                logits = model(images, return_logits=True)
            else:
                logits = torch.stack([
                    model(images, return_logits=True)
                    for _ in range(mc_samples)
                ]).mean(dim=0)
            total_loss += criterion(logits, labels).item() * images.size(0)
            total_samples += images.size(0)
    return total_loss / total_samples


def train_classifier(
    model, loader, test_loader, device, epochs, lr, name, mc_samples=1,
):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    test_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images, return_logits=True)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'{name} epoch {epoch + 1:03d}: loss={running_loss / len(loader):.4f}')
        test_loss = evaluate_loss(model, test_loader, device, mc_samples)
        test_losses.append(test_loss)
        print(f'{name} epoch {epoch + 1:03d}: test_loss={test_loss:.4f}')
    return test_losses


def save_test_losses(losses, text_filename, figure_filename):
    names = tuple(losses)
    epochs = range(1, len(next(iter(losses.values()))) + 1)
    with open(text_filename, 'w', encoding='utf-8') as output:
        output.write('epoch\t' + '\t'.join(names) + '\n')
        for epoch_index, epoch in enumerate(epochs):
            values = (losses[name][epoch_index] for name in names)
            output.write(
                str(epoch) + '\t' + '\t'.join(f'{value:.10g}' for value in values) + '\n'
            )

    dark2 = ('#1b9e77', '#d95f02', '#7570b3')
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for color, name in zip(dark2, names):
        ax.plot(epochs, losses[name], color=color, linewidth=2.2, label=name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Test cross-entropy loss')
    ax.set_xticks(list(epochs))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_filename, dpi=180)
    plt.close(fig)
    print(f'Saved test losses to {text_filename} and {figure_filename}')


def predict_probabilities(model, loader, device, mc_samples=1):
    model.eval()
    probabilities = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if mc_samples == 1:
                batch_probabilities = _model_probabilities(model, images)
            else:
                draws = [_model_probabilities(model, images) for _ in range(mc_samples)]
                batch_probabilities = torch.stack(draws).mean(dim=0)
            probabilities.append(batch_probabilities.cpu())
            targets.append(labels.cpu())
    return torch.cat(probabilities), torch.cat(targets)


def _model_probabilities(model, images):
    logits = model(images, return_logits=True)
    return torch.softmax(logits, dim=-1)


def confusion_matrix(probabilities, targets, n_classes):
    predictions = probabilities.argmax(dim=-1).to(torch.int64).view(-1)
    targets = targets.to(torch.int64).view(-1)
    matrix = torch.zeros((n_classes, n_classes), dtype=torch.int64)
    for target, prediction in zip(targets, predictions):
        matrix[target, prediction] += 1
    return matrix


def plot_confusion_matrices(matrices, titles, digits, filename):
    fig, axes = plt.subplots(1, len(matrices), figsize=(5 * len(matrices), 4.5))
    if len(matrices) == 1:
        axes = [axes]
    tick_labels = [str(digit) for digit in digits]
    for ax, matrix, title in zip(axes, matrices, titles):
        matrix_np = matrix.numpy()
        image = ax.imshow(matrix_np, cmap='Blues')
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_xticks(range(len(digits)), labels=tick_labels)
        ax.set_yticks(range(len(digits)), labels=tick_labels)
        for row in range(len(digits)):
            for col in range(len(digits)):
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


def collect_prnn_sample_diagnostics(model, loader, device):
    model.eval()
    images_all = []
    labels_all = []
    probabilities_all = []
    strain_paths_all = []
    length_scales_all = []
    sigma_f_all = []
    plastic_strain_history_all = []
    equivalent_plastic_strain_history_all = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            sigma_f = None
            if model.image_parametrization == 'gp':
                _, sigma_f = model.encode_gp_parameters(images)
            (
                probabilities,
                strain_paths,
                length_scales,
                _,
                plastic_strain_history,
                equivalent_plastic_strain_history,
            ) = model(
                images,
                return_paths=True,
                return_history=True,
            )
            if length_scales is None:
                length_scales = torch.full(
                    (images.size(0), 3),
                    float('nan'),
                    device=device,
                    dtype=strain_paths.dtype,
                )
            if sigma_f is None:
                sigma_f = torch.full_like(length_scales, float('nan'))
            elif sigma_f.dim() == 1:
                sigma_f = sigma_f.unsqueeze(0).expand(images.size(0), -1)

            images_all.append(images.cpu())
            labels_all.append(labels.cpu())
            probabilities_all.append(probabilities.cpu())
            strain_paths_all.append(strain_paths.cpu())
            length_scales_all.append(length_scales.cpu())
            sigma_f_all.append(sigma_f.cpu())
            plastic_strain_history_all.append(plastic_strain_history.cpu())
            equivalent_plastic_strain_history_all.append(
                equivalent_plastic_strain_history.cpu()
            )

    return {
        'images': torch.cat(images_all),
        'labels': torch.cat(labels_all),
        'probabilities': torch.cat(probabilities_all),
        'strain_paths': torch.cat(strain_paths_all),
        'length_scales': torch.cat(length_scales_all),
        'sigma_f': torch.cat(sigma_f_all),
        'plastic_strain_history': torch.cat(plastic_strain_history_all),
        'equivalent_plastic_strain_history': torch.cat(
            equivalent_plastic_strain_history_all
        ),
    }


def _padded_limits(min_values, max_values):
    padding = 0.05 * (max_values - min_values)
    padding = torch.where(
        padding == 0,
        torch.full_like(padding, 1e-12),
        padding,
    )
    return torch.stack((min_values - padding, max_values + padding), dim=-1)


def plot_prnn_sample_diagnostics(diagnostics, digits, filename):
    images = diagnostics['images']
    labels = diagnostics['labels']
    probabilities = diagnostics['probabilities']
    strain_paths = diagnostics['strain_paths']
    plastic_strain_history = diagnostics['plastic_strain_history']
    equivalent_plastic_strain_history = diagnostics[
        'equivalent_plastic_strain_history'
    ]
    length_scales = diagnostics['length_scales']
    sigma_f = diagnostics['sigma_f']
    n_samples = images.size(0)
    time_steps = np.arange(strain_paths.size(1))
    strain_names = (r'$\epsilon_{xx}$', r'$\epsilon_{yy}$', r'$\gamma_{xy}$')
    epsp_min = plastic_strain_history.amin(dim=(0, 1, 2))
    epsp_max = plastic_strain_history.amax(dim=(0, 1, 2))
    epsp_strain_min = torch.minimum(strain_paths.amin(dim=(0, 1)), epsp_min)
    epsp_strain_max = torch.maximum(strain_paths.amax(dim=(0, 1)), epsp_max)
    epsp_y_limits = _padded_limits(epsp_strain_min, epsp_strain_max)
    epspeq_y_limits = _padded_limits(
        equivalent_plastic_strain_history.amin(dim=(0, 1, 2)),
        equivalent_plastic_strain_history.amax(dim=(0, 1, 2)),
    )
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig = plt.figure(figsize=(16, max(3.2 * n_samples, 3.5)))
    grid = fig.add_gridspec(
        n_samples,
        5,
        width_ratios=(1.0, 2.2, 2.2, 2.2, 1.0),
        hspace=0.55,
        wspace=0.35,
    )
    for row in range(n_samples):
        image = images[row].squeeze().numpy()
        target_index = int(labels[row].item())
        predicted_index = int(probabilities[row].argmax().item())
        confidence = float(probabilities[row, predicted_index].item())
        target_digit = digits[target_index]
        predicted_digit = digits[predicted_index]
        is_correct = predicted_index == target_index

        ax_image = fig.add_subplot(grid[row, 0])
        ax_image.imshow(image, cmap='gray')
        ax_image.set_ylabel(f'Sample {row + 1}')
        ax_image.set_xticks([])
        ax_image.set_yticks([])
        if row == 0:
            ax_image.set_title('Input image')

        for component in range(3):
            subgrid = grid[row, component + 1].subgridspec(
                2,
                1,
                height_ratios=(1.0, 1.0),
                hspace=0.12,
            )
            ax_epsp = fig.add_subplot(subgrid[0])
            ax_epspeq = fig.add_subplot(subgrid[1], sharex=ax_epsp)
            ax_epsp.plot(
                time_steps,
                strain_paths[row, :, component].numpy(),
                color='0.35',
                linewidth=1.5,
            )
            for point in range(plastic_strain_history.size(2)):
                point_color = color_cycle[point % len(color_cycle)]
                ax_epsp.plot(
                    time_steps,
                    plastic_strain_history[row, :, point, component].numpy(),
                    alpha=0.35,
                    color=point_color,
                    linewidth=0.9,
                )
                ax_epspeq.plot(
                    time_steps,
                    equivalent_plastic_strain_history[row, :, point].numpy(),
                    alpha=0.35,
                    color=point_color,
                    linestyle='--',
                    linewidth=0.9,
                )
            ax_epsp.set_ylim(epsp_y_limits[component].tolist())
            ax_epspeq.set_ylim(epspeq_y_limits.tolist())
            ax_epsp.tick_params(labelbottom=False)
            length_scale = length_scales[row, component].item()
            sigma_f_value = sigma_f[row, component].item()
            if np.isnan(length_scale) or np.isnan(sigma_f_value):
                ax_epspeq.set_xlabel('timesignal pixel strain')
            else:
                ax_epspeq.set_xlabel(
                    rf'$\ell$={length_scale:.3g}, '
                    rf'$\sigma_f$={sigma_f_value:.3g}'
                )
            if row == 0:
                ax_epsp.set_title(strain_names[component])
            if component == 0:
                ax_epsp.set_ylabel('strain / epsp')
                ax_epspeq.set_ylabel('epspeq')

        ax_labels = fig.add_subplot(grid[row, 4])
        ax_labels.axis('off')
        label_color = 'tab:green' if is_correct else 'tab:red'
        ax_labels.text(
            0.5,
            0.5,
            f'pred: {predicted_digit}\n'
            f'target: {target_digit}\n'
            f'conf: {confidence:.2f}',
            ha='center',
            va='center',
            color=label_color,
            fontsize=11,
            transform=ax_labels.transAxes,
        )
        if row == 0:
            ax_labels.set_title('PRNN labels')

    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved PRNN sample diagnostics to {filename}')


def main():
    args = parse_args()
    args.digits = validate_digits(args.digits)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.get_default_dtype()
    n_classes = len(args.digits)

    train_dataset, test_dataset = load_mnist_subset(
        args.data_dir,
        args.digits,
        args.train_size,
        args.test_size,
        args.image_size,
        dtype,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    image_shape = train_dataset.tensors[0].shape[1:]

    common_prnn = dict(
        image_shape=image_shape,
        n_matpts=args.mat_pts,
        seq_len=args.seq_len,
        n_classes=n_classes,
        image_parametrization=args.image_parametrization,
        timesignal_smoothing_steps=args.timesignal_smoothing_steps,
        sigma_f_mode=args.sigma_f_mode,
        device=device,
    )
    linear_prnn = PRNNClassifier(
        **common_prnn,
        material_type='linear',
        decoder_features='stress',
    ).to(device)
    plastic_prnn = PRNNClassifier(
        **common_prnn,
        material_type='j2',
        decoder_features=args.decoder_features,
    ).to(device)
    mlp = MLPClassifier(
        image_shape=image_shape,
        n_classes=n_classes,
        hidden_sizes=(3 * args.mat_pts,),
        device=device,
    ).to(device)

    print(f'Training on MNIST digits {args.digits} ({n_classes} classes).')
    print(f'PRNN decoder features: {args.decoder_features}')
    print(f'PRNN image parametrization: {args.image_parametrization}')
    print(f'Timesignal smoothing steps: {args.timesignal_smoothing_steps}')
    print(f'PRNN sigma_f mode: {args.sigma_f_mode}')
    print(f'Hidden/material units per model: {3 * args.mat_pts}')
    losses = {}
    losses['Linear PRNN'] = train_classifier(
        linear_prnn, train_loader, test_loader, device, args.epochs, args.lr,
        'Linear PRNN', args.mc_samples,
    )
    losses['Plastic PRNN'] = train_classifier(
        plastic_prnn, train_loader, test_loader, device, args.epochs, args.lr,
        'Plastic PRNN', args.mc_samples,
    )
    losses['MLP'] = train_classifier(
        mlp, train_loader, test_loader, device, args.epochs, args.lr, 'MLP', 1,
    )
    save_test_losses(losses, args.loss_file, args.loss_figure)

    linear_probabilities, targets = predict_probabilities(
        linear_prnn,
        test_loader,
        device,
        args.mc_samples,
    )
    plastic_probabilities, _ = predict_probabilities(
        plastic_prnn, test_loader, device, args.mc_samples,
    )
    mlp_probabilities, _ = predict_probabilities(mlp, test_loader, device, 1)
    linear_matrix = confusion_matrix(linear_probabilities, targets, n_classes)
    plastic_matrix = confusion_matrix(plastic_probabilities, targets, n_classes)
    mlp_matrix = confusion_matrix(mlp_probabilities, targets, n_classes)

    print('Linear PRNN confusion matrix:\n', linear_matrix.numpy())
    print('Plastic PRNN confusion matrix:\n', plastic_matrix.numpy())
    print('MLP confusion matrix:\n', mlp_matrix.numpy())
    plot_confusion_matrices(
        [linear_matrix, plastic_matrix, mlp_matrix],
        [
            f'Linear PRNN ({args.mc_samples} MC)',
            f'Plastic PRNN ({args.mc_samples} MC)',
            'MLP baseline',
        ],
        args.digits,
        args.figure,
    )
    diagnostics = collect_prnn_sample_diagnostics(
        plastic_prnn, test_loader, device,
    )
    plot_prnn_sample_diagnostics(diagnostics, args.digits, args.sample_figure)


if __name__ == '__main__':
    main()
