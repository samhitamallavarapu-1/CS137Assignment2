import argparse
from pathlib import Path

import torch

from model import get_model


def load_metadata(metadata_path: Path) -> dict:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    return torch.load(metadata_path, weights_only=False)


def build_dummy_input(batch_size: int, height: int, width: int, n_channels: int) -> torch.Tensor:
    # Your model expects channels-last input: (B, H, W, C)
    return torch.randn(batch_size, height, width, n_channels)


def print_model_info(model: torch.nn.Module, dummy_input: torch.Tensor) -> None:
    print("\n=== MODEL ===")
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n=== PARAM COUNTS ===")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    try:
        from torchinfo import summary

        print("\n=== TORCHINFO SUMMARY ===")
        summary(
            model,
            input_data=dummy_input,
            depth=4,
            col_names=("input_size", "output_size", "num_params", "trainable"),
            verbose=1,
        )
    except ImportError:
        print("\n[INFO] torchinfo not installed; skipping detailed summary.")
        print("Install with: pip install torchinfo")


def save_graph(model: torch.nn.Module, dummy_input: torch.Tensor, output_prefix: Path) -> None:
    try:
        from torchviz import make_dot
    except ImportError:
        print("\n[INFO] torchviz not installed; skipping graph rendering.")
        print("Install with: pip install torchviz")
        return

    model.eval()
    output = model(dummy_input)

    dot = make_dot(
        output,
        params=dict(model.named_parameters()),
        show_attrs=False,
        show_saved=False,
    )

    dot.format = "png"
    rendered_path = dot.render(str(output_prefix), cleanup=True)
    print(f"\nSaved architecture graph to: {rendered_path}")


def save_text_architecture(model: torch.nn.Module, output_path: Path) -> None:
    with open(output_path, "w") as f:
        f.write(str(model))
        f.write("\n\n")
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        f.write(f"Total parameters: {total_params:,}\n")
        f.write(f"Trainable parameters: {trainable_params:,}\n")

    print(f"Saved text architecture to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize WeatherCNN architecture")
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=Path("/cluster/tufts/c26sp1cs0137/data/assignment2_data/dataset/metadata.pt"),
        help="Path to metadata.pt",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=450)
    parser.add_argument("--width", type=int, default=449)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_visualization"),
        help="Directory to save outputs",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.metadata_path)
    model = get_model(metadata)

    n_channels = int(metadata.get("n_vars", len(metadata.get("variable_names", []))))
    dummy_input = build_dummy_input(
        batch_size=args.batch_size,
        height=args.height,
        width=args.width,
        n_channels=n_channels,
    )

    print_model_info(model, dummy_input)
    save_text_architecture(model, args.output_dir / "model_architecture.txt")
    save_graph(model, dummy_input, args.output_dir / "weathercnn_graph")


if __name__ == "__main__":
    main()