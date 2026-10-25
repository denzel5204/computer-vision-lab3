import copy
import random
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data import CLASS_NAMES, load_data
from model import CarTruckCNN


ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = ROOT / "images"
RANDOM_STATE = 42
BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.001


def set_random_state(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels.float())

            if training:
                loss.backward()
                optimizer.step()

        predictions = (logits >= 0).long()
        total_loss += loss.item() * len(labels)
        correct += (predictions == labels).sum().item()
        total += len(labels)

    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    labels_all = []
    probabilities_all = []
    correct_examples = []
    error_examples = []

    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            probabilities = torch.sigmoid(logits).cpu()
            predictions = (probabilities >= 0.5).long()
            loss = criterion(logits, labels.to(device).float())

            total_loss += loss.item() * len(labels)
            labels_all.append(labels)
            probabilities_all.append(probabilities)

            for image, label, prediction, probability in zip(
                images, labels, predictions, probabilities
            ):
                example = (image, int(label), int(prediction), float(probability))
                target = correct_examples if label == prediction else error_examples
                if len(target) < 4:
                    target.append(example)

    labels = torch.cat(labels_all).numpy()
    probabilities = torch.cat(probabilities_all).numpy()
    predictions = (probabilities >= 0.5).astype(int)
    metrics, matrix = calculate_metrics(labels, predictions)

    return (
        total_loss / len(labels),
        metrics,
        matrix,
        correct_examples + error_examples,
    )


def calculate_metrics(labels, predictions):
    true_negative = int(np.sum((labels == 0) & (predictions == 0)))
    false_positive = int(np.sum((labels == 0) & (predictions == 1)))
    false_negative = int(np.sum((labels == 1) & (predictions == 0)))
    true_positive = int(np.sum((labels == 1) & (predictions == 1)))

    accuracy = (true_positive + true_negative) / len(labels)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
    matrix = np.array(
        [[true_negative, false_positive], [false_negative, true_positive]]
    )
    return metrics, matrix


def plot_training(history, best_epoch):
    indexes = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(indexes, history["train_loss"], marker="o", label="Обучение")
    axes[0].plot(indexes, history["validation_loss"], marker="o", label="Проверка")
    axes[0].set_title("Изменение функции потерь")
    axes[0].set_xlabel("Эпоха")
    axes[0].set_ylabel("BCEWithLogitsLoss")

    axes[1].plot(indexes, history["train_accuracy"], marker="o", label="Обучение")
    axes[1].plot(
        indexes,
        history["validation_accuracy"],
        marker="o",
        label="Проверка",
    )
    axes[1].axvline(best_epoch, color="#8d739d", linestyle="--", label="Лучшая эпоха")
    axes[1].set_title("Изменение точности")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylabel("Accuracy")

    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_axisbelow(True)
        axis.legend()

    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    image = ax.imshow(matrix, cmap="Blues")
    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
                fontsize=15,
                color="white" if matrix[row, column] > matrix.max() / 2 else "#1f2937",
            )
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Настоящий класс")
    ax.set_title("Матрица ошибок на тестовой выборке")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "confusion_matrix.png", dpi=180)
    plt.close(fig)


def plot_predictions(examples):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
    for axis, (image, label, prediction, probability) in zip(axes.flat, examples):
        shown = (image * 0.5 + 0.5).permute(1, 2, 0).numpy()
        axis.imshow(np.clip(shown, 0, 1))
        axis.set_title(
            f"Ответ: {CLASS_NAMES[label]}\n"
            f"Прогноз: {CLASS_NAMES[prediction]} ({probability:.2f})",
            color="#2f7773" if label == prediction else "#b24a4a",
            fontsize=9,
        )
        axis.axis("off")
    fig.suptitle("Примеры правильных и ошибочных предсказаний")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "predictions.png", dpi=180)
    plt.close(fig)


def main():
    set_random_state(RANDOM_STATE)
    IMAGES_DIR.mkdir(exist_ok=True)
    device = select_device()
    loaders, sizes = load_data(BATCH_SIZE, RANDOM_STATE)

    model = CarTruckCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {
        "train_loss": [],
        "validation_loss": [],
        "train_accuracy": [],
        "validation_accuracy": [],
    }
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None
    started_at = perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = run_epoch(
            model, loaders["train"], criterion, device, optimizer
        )
        validation_loss, validation_accuracy = run_epoch(
            model, loaders["validation"], criterion, device
        )

        history["train_loss"].append(train_loss)
        history["validation_loss"].append(validation_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_accuracy"].append(validation_accuracy)

        if validation_accuracy > best_accuracy:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Эпоха {epoch:02d}/{EPOCHS}: "
            f"loss={train_loss:.4f}, accuracy={train_accuracy:.4f}, "
            f"val_loss={validation_loss:.4f}, val_accuracy={validation_accuracy:.4f}"
        )

    training_seconds = perf_counter() - started_at
    model.load_state_dict(best_state)
    test_loss, metrics, matrix, examples = evaluate(
        model, loaders["test"], criterion, device
    )

    plot_training(history, best_epoch)
    plot_confusion_matrix(matrix)
    plot_predictions(examples)

    best_index = best_epoch - 1
    gap = (
        history["train_accuracy"][best_index]
        - history["validation_accuracy"][best_index]
    )

    print("\nИтоговый результат")
    print(f"Устройство: {device}")
    print(f"Размеры выборок: {sizes}")
    print(f"Лучшая эпоха: {best_epoch}")
    print(f"Время обучения: {training_seconds:.1f} с")
    print(f"Validation accuracy: {best_accuracy:.4f}")
    print(f"Разница train и validation accuracy: {gap:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    for name, value in metrics.items():
        print(f"Test {name}: {value:.4f}")
    print(f"Матрица ошибок: {matrix.tolist()}")


if __name__ == "__main__":
    main()
