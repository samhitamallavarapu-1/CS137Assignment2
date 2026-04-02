from pathlib import Path

import scipy.io
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class StanfordCarsDataset(Dataset):
    """
    Stanford Cars dataset loader for the official dataset structure.

    Expected folder layout:
    project_root/
    └── stanford_cars/
        ├── cars_train/
        ├── cars_test/
        ├── cars_test_annos_withlabels.mat
        └── devkit/
            └── devkit/
                ├── cars_train_annos.mat
                ├── cars_meta.mat
                └── ...
    """

    def __init__(self, root: Path, split: str = "train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # your files appear to be nested one extra level deep
        self.devkit_dir = self.root / "devkit" / "devkit"

        self.class_names = self._load_class_names()

        if split == "train":
            annos_path = self.devkit_dir / "cars_train_annos.mat"
            image_dir = self.root / "cars_train"
            self.samples = self._load_train_annos(annos_path, image_dir)

        elif split == "test":
            annos_path = self.root / "cars_test_annos_withlabels.mat"
            image_dir = self.root / "cars_test"
            self.samples = self._load_test_annos(annos_path, image_dir)

        else:
            raise ValueError("split must be 'train' or 'test'")

    def _load_class_names(self):
        meta_path = self.devkit_dir / "cars_meta.mat"
        meta = scipy.io.loadmat(meta_path)
        class_names_raw = meta["class_names"][0]
        class_names = [str(x[0]) for x in class_names_raw]
        return class_names

    def _load_train_annos(self, annos_path: Path, image_dir: Path):
        data = scipy.io.loadmat(annos_path, squeeze_me=True)
        annos = data["annotations"]

        samples = []
        for anno in annos:
            label = int(anno["class"]) - 1  # convert 1-indexed to 0-indexed
            fname = str(anno["fname"])
            img_path = image_dir / fname

            samples.append({
                "image_path": img_path,
                "label": label,
                "class_name": self.class_names[label],
            })

        return samples

    def _load_test_annos(self, annos_path: Path, image_dir: Path):
        data = scipy.io.loadmat(annos_path, squeeze_me=True)
        annos = data["annotations"]

        samples = []
        for anno in annos:
            label = int(anno["class"]) - 1
            fname = str(anno["fname"])
            img_path = image_dir / fname

            samples.append({
                "image_path": img_path,
                "label": label,
                "class_name": self.class_names[label],
            })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        label = sample["label"]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_cars_transforms(image_size: int = 224):
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform


def get_cars_datasets(root: Path, image_size: int = 224):
    train_transform, eval_transform = get_cars_transforms(image_size=image_size)

    train_dataset = StanfordCarsDataset(
        root=root,
        split="train",
        transform=train_transform,
    )

    test_dataset = StanfordCarsDataset(
        root=root,
        split="test",
        transform=eval_transform,
    )

    return train_dataset, test_dataset


def get_cars_dataloaders(
    root: Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
):
    train_dataset, test_dataset = get_cars_datasets(root=root, image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader, train_dataset, test_dataset