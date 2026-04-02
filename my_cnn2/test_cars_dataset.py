from pathlib import Path
from cars_dataset import get_cars_datasets

CARS_ROOT = Path(__file__).resolve().parent / "stanford_cars"

train_dataset, test_dataset = get_cars_datasets(CARS_ROOT, image_size=224)

print("Train size:", len(train_dataset))
print("Test size:", len(test_dataset))
print("Num classes:", len(train_dataset.class_names))

x, y = train_dataset[0]
print("Sample image shape:", x.shape)
print("Sample label:", y)
print("Sample class:", train_dataset.class_names[y])