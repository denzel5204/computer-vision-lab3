from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


CLASS_NAMES = ["Автомобиль", "Грузовик"]
SOURCE_CLASSES = {1: 0, 9: 1}
DATA_DIR = Path.home() / ".cache" / "computer-vision-lab3"


class BinaryCIFAR10(Dataset):
    def __init__(self, dataset, indexes, transform):
        self.dataset = dataset
        self.indexes = indexes
        self.transform = transform

    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, position):
        image, source_class = self.dataset[self.indexes[position]]
        return self.transform(image), SOURCE_CLASSES[source_class]


def _indexes_for_classes(targets, generator):
    indexes = {}
    for source_class in SOURCE_CLASSES:
        selected = torch.tensor(
            [index for index, value in enumerate(targets) if value == source_class]
        )
        indexes[source_class] = selected[
            torch.randperm(len(selected), generator=generator)
        ]
    return indexes


def _shuffle(indexes, generator):
    order = torch.randperm(len(indexes), generator=generator)
    return indexes[order].tolist()


def load_data(batch_size=128, random_state=42):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(random_state)

    source_train = datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    source_test = datasets.CIFAR10(root=DATA_DIR, train=False, download=True)

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Из каждого класса берём 4000 изображений для обучения и 1000 для проверки
    class_indexes = _indexes_for_classes(source_train.targets, generator)
    train_indexes = torch.cat([values[:4000] for values in class_indexes.values()])
    validation_indexes = torch.cat([values[4000:] for values in class_indexes.values()])

    test_indexes = [
        index
        for index, value in enumerate(source_test.targets)
        if value in SOURCE_CLASSES
    ]

    datasets_by_part = {
        "train": BinaryCIFAR10(
            source_train,
            _shuffle(train_indexes, generator),
            train_transform,
        ),
        "validation": BinaryCIFAR10(
            source_train,
            _shuffle(validation_indexes, generator),
            evaluation_transform,
        ),
        "test": BinaryCIFAR10(source_test, test_indexes, evaluation_transform),
    }

    loaders = {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=0,
        )
        for name, dataset in datasets_by_part.items()
    }

    return loaders, {name: len(dataset) for name, dataset in datasets_by_part.items()}
