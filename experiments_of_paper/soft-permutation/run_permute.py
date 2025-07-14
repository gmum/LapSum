import argparse
import json
import pickle
import random
from math import log
from pathlib import Path
from time import time

import torch
import torch.optim as optim
from tqdm import tqdm

from lapsum.permute import cross_entropy
import multi_mnist_cnn
from mnist_input import get_iterators, transform_target


class CrossEntropyLoss(torch.nn.Module):
    def __init__(
            self, alpha: float, alpha_trainable: bool = False, reduction='mean', perturbed_alpha=False,
            noise_type='uniform'
    ):
        super(CrossEntropyLoss, self).__init__()
        assert alpha > 0, "alpha must be positive"
        self.reduction = reduction

        self.alpha = torch.nn.Parameter(torch.tensor(log(alpha))) if alpha_trainable \
            else torch.tensor(alpha, requires_grad=False)
        self.alpha_trainable = alpha_trainable
        self.perturb_alpha = perturbed_alpha

        if noise_type == 'uniform':
            self.noise_function = CrossEntropyLoss.add_probabilistic_noise
        elif noise_type == 'gumbel':
            self.noise_function = CrossEntropyLoss.add_gumbel_noise
        else:
            raise ValueError(f"Invalid noise type: {noise_type}. You can choose between: 'uniform' or 'gumbel'.")

    def forward(self, x, target, current_iter=1, num_iters=1, noise_prob=0, noise_scale=0):
        alpha = torch.exp(self.alpha) if self.alpha_trainable else self.alpha
        if self.perturb_alpha:
            alpha = CrossEntropyLoss.add_probabilistic_noise(alpha, current_iter, num_iters, noise_scale, noise_prob)

        x = self.noise_function(x, current_iter, num_iters, noise_scale, noise_prob)
        return cross_entropy(x, target, alpha, reduction=self.reduction)

    @staticmethod
    def add_probabilistic_noise(
            value: torch.Tensor, current_iter: int, num_iters: int, init_factor: float = 1, noise_prob: float = 0.5
    ):
        if current_iter >= num_iters and random.random() > noise_prob:
            return value
        noise_factor = init_factor * (1 - (current_iter / num_iters))
        noise = torch.rand_like(value) * 2 * noise_factor - noise_factor
        return value + noise

    @staticmethod
    def add_gumbel_noise(
            value: torch.Tensor, current_iter: int, num_iters: int, init_factor: float = 1, noise_prob: float = 0.5
    ):
        if current_iter >= num_iters and random.random() > noise_prob:
            return value
        noise_factor = init_factor * (1 - (current_iter / num_iters))
        gumbels = -torch.empty_like(value).exponential_().log()  # ~Gumbel(0,1)
        return value + gumbels * noise_factor
        # tau = max(0.2, init_factor * (1 - (current_iter / num_iters)))
        # gumbels = (value + gumbels) / tau  # ~Gumbel(logits,tau)
        # return gumbels


def save_dict(path2save, dict2save):
    with open(path2save, 'wb') as f:
        pickle.dump(dict2save, f)


def load_dict(path2dict):
    with open(path2dict, 'rb') as f:
        loaded_dict = pickle.load(f)
    return loaded_dict


def get_parameters():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a sorting model on MNIST data.')
    parser.add_argument('--root_data', type=str, default='MNIST_data', help='Path to the MNIST data directory')
    parser.add_argument('--root_save', type=str, required=True, help='Path to save the model and results')
    parser.add_argument('--M', type=int, default=1, help='Batch size')
    parser.add_argument('--n', type=int, default=3, help='Number of elements to compare at a time')
    parser.add_argument('--l', type=int, default=4, help='Number of digits in each sequence')
    parser.add_argument('--alpha', type=float, default=1, help='')
    parser.add_argument('--alpha_trainable', action='store_true', help='Whether alpha is trainable')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--perturbed_alpha', action='store_true', help='Whether to perturb alpha')
    parser.add_argument('--noise_prob', type=float, default=0, help='Probability of adding noise')
    parser.add_argument('--noise_scale', type=float, default=0, help='Scale of the noise')
    parser.add_argument('--noise_type', choices=['uniform', 'gumbel'], default='uniform', help='Type of noise to add')
    parser.add_argument('--test-only', action='store_true', help='Only test the model')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume training from')
    return parser


def save_checkpoint(path, model, optimizer, criterion, logger_data, epoch):
    """Save model, optimizer, criterion, logger_data, and epoch to a checkpoint file."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'criterion_state_dict': criterion.state_dict(),
        'logger_data': logger_data,
    }, path)


def load_checkpoint(path, model, optimizer, criterion):
    """Load model, optimizer, criterion, logger_data, and epoch from a checkpoint file."""
    checkpoint = torch.load(path, weights_only=True)
    if model:
        model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if criterion:
        criterion.load_state_dict(checkpoint['criterion_state_dict'])
    return checkpoint['logger_data'], checkpoint['epoch']


def train_model(model, criterion, optimizer, train_iterator, device, args, logger_data, epoch):
    model.train()
    logger_loss = 0
    start_time = time()

    iter_val = (epoch - 1) * len(train_iterator)
    with tqdm(total=len(train_iterator), desc=f"Train {epoch}/{args.num_epochs}", leave=False) as train_pbar:
        for data, _, _, labels in train_iterator:
            data = data.to(device)
            target = transform_target(labels).to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target, iter_val, args.num_iters, args.noise_prob, args.noise_scale)
            loss.backward()
            optimizer.step()

            logger_loss += loss.item()
            train_pbar.update(1)
            train_pbar.set_postfix(loss=loss.item())
            if args.alpha_trainable:
                logger_data["alpha"].append(criterion.alpha.exp().item())
            iter_val += 1

    logger_data["train_times"].append(time() - start_time)
    logger_data["train_loss"].append(logger_loss / len(train_iterator))


def validate_model(model, criterion, val_iterator, device, logger_data, epoch):
    model.eval()

    logger_loss = 0
    any_correct = []
    all_correct = []
    start_time = time()

    with tqdm(total=len(val_iterator), desc=f"Validation {epoch}", leave=False) as val_pbar:
        with torch.no_grad():
            for data, _, _, labels in val_iterator:
                data = data.to(device)
                target = transform_target(labels).to(device)

                output = model(data)
                loss = criterion(output, target)
                logger_loss += loss.item()

                # Calculate accuracy
                pred = transform_target(output)
                any_correct.append((pred == target).float().mean().item())
                all_correct.append((pred == target).all(dim=-1).float().mean().item())

                val_pbar.update(1)
                val_pbar.set_postfix(loss=loss.item(), allC=all_correct[-1], anyC=any_correct[-1])

    logger_data["val_times"].append(time() - start_time)
    logger_data["val_loss"].append(logger_loss / len(val_iterator))
    logger_data["val_all_correct"].append(sum(all_correct) / len(all_correct))
    logger_data["val_any_correct"].append(sum(any_correct) / len(any_correct))


def test_model(model, test_iterator, device):
    """Test the model and return accuracy metrics."""
    model.eval()
    any_correct, all_correct = [], []

    with torch.no_grad():
        for data, _, _, labels in tqdm(test_iterator, desc="Testing"):
            data = data.to(device)
            target = transform_target(labels).to(device)

            output = model(data)

            # Calculate accuracy
            pred = transform_target(output)
            any_correct.append((pred == target).float().mean().item())
            all_correct.append((pred == target).all(dim=-1).float().mean().item())

    test_all_correct = sum(all_correct) / len(test_iterator)
    test_any_correct = sum(any_correct) / len(test_iterator)
    return test_all_correct, test_any_correct


def main():
    args = get_parameters().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\033[0;1;31m{device=}\033[0m")
    print(args)

    # Initialize data iterators
    train_iterator, val_iterator, test_iterator = get_iterators(
        args.l, args.n, 10 ** args.l - 1, minibatch_size=args.M,
        path2data=args.root_data, download_data=False, num_workers=args.workers, seed=args.seed
    )

    args.root_save = Path(args.root_save)
    args.root_save.mkdir(parents=True, exist_ok=True)
    save_dict(args.root_save / "params.pkl", vars(args))

    # Instantiate the model
    model = multi_mnist_cnn.DeepNN(args.l, 1)
    model.to(device)

    # Dictionary to store all training data
    logger_data = {
        "train_loss": [],
        "val_loss": [],
        "val_all_correct": [],
        "val_any_correct": [],
        "train_times": [],
        "val_times": [],
        "best_accuracy": 0,
        "test_all_correct": None,
        "test_any_correct": None,
        "alpha": [] if args.alpha_trainable else args.alpha,
    }

    # Test-only mode
    if args.test_only:
        model_path = args.root_save / "model.pth"
        if model_path.is_file():
            print(f"Testing model from checkpoint: {args.resume}")
            model.load_state_dict(torch.load(model_path, weights_only=True))
            test_all_correct, test_any_correct = test_model(model, test_iterator, device)
            print(
                f"Test All Correct: {test_all_correct * 100:.2f}%, "
                f"Test Any Correct: {test_any_correct * 100:.2f}%"
            )
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}.")

        if args.resume and Path(args.resume).is_file():
            logger_data, _ = load_checkpoint(args.resume, None, None, None)
            logger_data["test_all_correct"] = test_all_correct
            logger_data["test_any_correct"] = test_any_correct

            # Save the training data to a JSON file
            filename = args.root_save / "results.json"
            filename.write_text(json.dumps(logger_data, sort_keys=True, indent=4), encoding="utf-8")

            # Delete checkpoint file
            Path(args.resume).unlink()
            print(f"Deleted checkpoint file: {args.resume}")
        return

    # Loss function and optimizer
    criterion = CrossEntropyLoss(args.alpha, args.alpha_trainable, perturbed_alpha=args.perturbed_alpha,
                                 noise_type=args.noise_type)
    criterion.to(device)
    parameters_fit = list(model.parameters()) + (list(criterion.parameters()) if args.alpha_trainable else [])
    print(f"Number of parameters: {sum(p.numel() for p in parameters_fit)}")
    optimizer = optim.Adam(parameters_fit, lr=args.lr)

    start_epoch = 1
    if args.resume and Path(args.resume).is_file():
        print(f"Resuming training from checkpoint: {args.resume}")
        logger_data, start_epoch = load_checkpoint(args.resume, model, optimizer, criterion)
        start_epoch += 1  # Start from the next epoch

    args.num_iters = int(0.9 * args.num_epochs * len(train_iterator))

    # Global tqdm progress bar for the entire training process
    with tqdm(total=args.num_epochs, desc="Training Progress", initial=start_epoch - 1) as global_pbar:
        for epoch in range(start_epoch, args.num_epochs + 1):
            ####################################################
            #                 Train Step
            ####################################################

            train_model(model, criterion, optimizer, train_iterator, device, args, logger_data, epoch)

            ####################################################
            #                  Validation Step
            ####################################################
            validate_model(model, criterion, val_iterator, device, logger_data, epoch)

            # Update global progress bar
            global_pbar.update(1)
            global_pbar.set_postfix({
                "TLoss": f"{logger_data['train_loss'][-1]:.2f}",
                "VLoss": f"{logger_data['val_loss'][-1]:.2f}",
                "allC": f"{logger_data['val_all_correct'][-1] * 100:.2f}%",
                "anyC": f"{logger_data['val_any_correct'][-1] * 100:.2f}%",
                "TTime": f"{logger_data['train_times'][-1]:.2f}s",
                "VTime": f"{logger_data['val_times'][-1]:.2f}s"
            })

            ####################################################
            #                 Save Best Model
            ####################################################
            if logger_data["val_all_correct"][-1] > logger_data["best_accuracy"]:
                logger_data["best_accuracy"] = logger_data["val_all_correct"][-1]
                torch.save(model.state_dict(), args.root_save / "model.pth")
                print(f"New best model saved with accuracy: {logger_data['best_accuracy'] * 100:.2f}%")

            ####################################################
            #                 Save Checkpoint
            ####################################################
            save_checkpoint(args.root_save / "checkpoint.pth", model, optimizer, criterion, logger_data, epoch)

    if not args.alpha_trainable:  # todo: remove, check whether alpha was changed
        logger_data["alpha"] = criterion.alpha.item()

    ####################################################
    #                   Test Step
    ####################################################
    # Load the best model
    if (args.root_save / "model.pth").is_file():
        model.load_state_dict(torch.load(args.root_save / "model.pth", weights_only=True))
        test_all_correct, test_any_correct = test_model(model, test_iterator, device)
        logger_data["test_all_correct"] = test_all_correct
        logger_data["test_any_correct"] = test_any_correct
        print(
            f"Test All Correct: {logger_data['test_all_correct'] * 100:.2f}%, "
            f"Test Any Correct: {logger_data['test_any_correct'] * 100:.2f}%"
        )

        ####################################################
        #                   Save Training Data
        ####################################################
        # Save the training data to a JSON file
        filename = args.root_save / "results.json"
        filename.write_text(json.dumps(logger_data, sort_keys=True, indent=4), encoding="utf-8")

        print(f"Training data saved to '{filename}'.")

    # Delete checkpoint file
    Path(args.root_save / "checkpoint.pth").unlink(missing_ok=True)
    print("Program ended successfully.")


if __name__ == '__main__':
    main()
