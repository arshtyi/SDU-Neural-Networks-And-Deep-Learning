import csv
import gzip
import hashlib
import json
import platform
import struct
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from urllib.request import urlopen

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR = OUTPUT_DIR / "data"
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
RESOURCES = (
    ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
    ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
    ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
    ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c"),
)


@dataclass(frozen=True)
class Config:
    name: str
    architecture: str = "lenet"
    hidden_sizes: tuple[int, ...] = (256,)
    learning_rate: float = 0.001
    batch_size: int = 128
    epochs: int = 12


CONFIGS = (
    Config("mlp", architecture="mlp"),
    Config("mlp_deep", architecture="mlp", hidden_sizes=(256, 128)),
    Config("lenet"),
    Config("lenet_short", epochs=5),
    Config("lenet_lr_small", learning_rate=0.0003),
    Config("lenet_batch256", batch_size=256),
)


class LeNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(400, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))

    def feature_maps(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        first = self.conv1(x).relu()
        last = self.conv2(self.pool(first)).relu()
        return first, last

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, last = self.feature_maps(x)
        return self.classifier(self.pool(last))


def build_model(config: Config) -> nn.Module:
    if config.architecture == "lenet":
        return LeNet()
    layers: list[nn.Module] = [nn.Flatten()]
    in_features = 784
    for width in config.hidden_sizes:
        layers.extend((nn.Linear(in_features, width), nn.ReLU()))
        in_features = width
    layers.append(nn.Linear(in_features, 10))
    return nn.Sequential(*layers)


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    arrays = []
    for filename, checksum in RESOURCES:
        path = DATA_DIR / filename
        data = path.read_bytes() if path.exists() else b""
        if hashlib.md5(data).hexdigest() != checksum:
            print(f"下载 {filename}", flush=True)
            with urlopen(MNIST_URL + filename, timeout=60) as response:
                data = response.read()
            if hashlib.md5(data).hexdigest() != checksum:
                raise ValueError(f"MNIST 文件校验失败：{filename}")
            path.write_bytes(data)
        raw = gzip.decompress(data)
        magic, count = struct.unpack(">II", raw[:8])
        if magic == 2051:
            height, width = struct.unpack(">II", raw[8:16])
            if (height, width) != (28, 28):
                raise ValueError(f"MNIST 图像尺寸异常：{filename}")
            arrays.append(torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=16).copy()).reshape(count, 1, height, width).float().div_(255))
        elif magic == 2049:
            arrays.append(torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=8).copy()).long().reshape(count))
        else:
            raise ValueError(f"IDX 文件头异常：{filename}")
    x_train, y_train, x_test, y_test = arrays
    if len(y_train) != 60000 or len(y_test) != 10000 or len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("MNIST 样本数量异常。")
    return x_train, y_train, x_test, y_test


@torch.inference_mode()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> tuple[dict, torch.Tensor]:
    model.eval()
    logits = torch.cat([model(batch) for batch in x.split(512)])
    loss = nn.functional.cross_entropy(logits, y).item()
    correct = int((logits.argmax(dim=1) == y).sum().item())
    return {"loss": loss, "accuracy": correct / len(y), "correct": correct}, logits.softmax(dim=1)


def validation_key(row: dict) -> tuple[float, float]:
    return -row["validation_accuracy"], row["validation_loss"]


def train(config: Config, x: torch.Tensor, y: torch.Tensor, x_validation: torch.Tensor, y_validation: torch.Tensor) -> tuple[dict, list[dict]]:
    torch.manual_seed(RANDOM_STATE)
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    best: dict = {}
    start = perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        for indices in torch.randperm(len(y), generator=generator).split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
        train_metrics, _ = evaluate(model, x, y)
        validation_metrics, _ = evaluate(model, x_validation, y_validation)
        row = {"epoch": epoch, "train_loss": train_metrics["loss"], "train_accuracy": train_metrics["accuracy"]}
        row.update({"validation_loss": validation_metrics["loss"], "validation_accuracy": validation_metrics["accuracy"]})
        history.append(row)
        if not best or validation_key(row) < validation_key(best):
            best = row.copy()
            torch.save(
                {"config": asdict(config), "state_dict": model.state_dict(), "best": best, "preprocessing": "float32(N,1,28,28) / 255"},
                OUTPUT_DIR / "models" / f"{config.name}.pt",
            )
        print(
            f"{config.name:15s} {epoch:2d}/{config.epochs}: train loss={row['train_loss']:.4f}, acc={row['train_accuracy']:.4%}; "
            f"validation loss={row['validation_loss']:.4f}, acc={row['validation_accuracy']:.4%}",
            flush=True,
        )
    return {
        **asdict(config),
        "parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "best": best,
        "last": history[-1],
        "training_seconds": perf_counter() - start,
    }, history


def write_csv(filename: str, rows: list[dict]) -> None:
    with (OUTPUT_DIR / filename).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(width: float, height: float) -> Figure:
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    return figure


def save_curves(histories: dict[str, list[dict]], names: list[str], filename: str) -> None:
    figure = make_figure(10, 6)
    axes = figure.subplots(2, 2)
    for row, partition in enumerate(("train", "validation")):
        for column, metric in enumerate(("loss", "accuracy")):
            axis = axes[row, column]
            for name in names:
                history = histories[name]
                axis.plot([item["epoch"] for item in history], [item[f"{partition}_{metric}"] for item in history], marker=".", label=name)
            axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"{partition.capitalize()} {metric}")
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.savefig(OUTPUT_DIR / filename, dpi=180)


def save_confusion_matrices(results: list[dict], selected_names: dict[str, str]) -> None:
    figure = make_figure(11, 4.8)
    for axis, name in zip(figure.subplots(1, 2), selected_names.values(), strict=True):
        matrix = np.asarray(next(result for result in results if result["name"] == name)["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0)
        axis.set(xticks=range(10), yticks=range(10), xlabel="Predicted label", ylabel="True label", title=name)
        for row in range(10):
            for column in range(10):
                axis.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if matrix[row, column] > matrix.max() / 2 else "black",
                )
    figure.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=180)


@torch.inference_mode()
def save_features(model: LeNet, sample: torch.Tensor, sample_index: int, label: int) -> dict:
    model.eval()
    first, last = (feature[0].numpy() for feature in model.feature_maps(sample))
    kernels1 = model.conv1.weight.detach().numpy()[:, 0]
    kernels2 = model.conv2.weight.detach().numpy()
    prediction = int(model(sample).argmax(dim=1).item())
    figure = make_figure(10, 3.7)
    axes = figure.subplots(2, 7)
    axes[0, 0].imshow(sample[0, 0], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(f"Input #{sample_index}\ntrue={label}, pred={prediction}", fontsize=9)
    axes[1, 0].text(0.5, 0.5, "Top: C1 weights\nBottom: ReLU maps", ha="center", va="center", fontsize=9)
    for channel in range(6):
        axes[0, channel + 1].imshow(kernels1[channel], cmap="coolwarm", vmin=-abs(kernels1).max(), vmax=abs(kernels1).max())
        axes[1, channel + 1].imshow(first[channel], cmap="viridis", vmin=0, vmax=first.max())
        axes[0, channel + 1].set_title(f"C1 out {channel}", fontsize=9)
    for axis in axes.flat:
        axis.set_axis_off()
    figure.savefig(OUTPUT_DIR / "conv1_features.png", dpi=200)

    figure = make_figure(14, 5.5)
    axes = figure.subplots(6, 16)
    for input_channel in range(6):
        for output_channel in range(16):
            axis = axes[input_channel, output_channel]
            axis.imshow(kernels2[output_channel, input_channel], cmap="coolwarm", vmin=-abs(kernels2).max(), vmax=abs(kernels2).max())
            axis.set(xticks=[], yticks=[])
            if input_channel == 0:
                axis.set_title(f"out {output_channel}", fontsize=8)
            if output_channel == 0:
                axis.set_ylabel(f"in {input_channel}", fontsize=8)
    figure.suptitle("C3 weights: all 16 x 6 kernel slices (5 x 5 each)")
    figure.savefig(OUTPUT_DIR / "conv2_kernels.png", dpi=200)

    figure = make_figure(8, 6)
    for channel, axis in enumerate(figure.subplots(4, 4).flat):
        axis.imshow(last[channel], cmap="viridis", vmin=0, vmax=last.max())
        axis.set_title(f"C3 out {channel}", fontsize=9)
        axis.set_axis_off()
    figure.suptitle("C3 ReLU feature maps: 16 channels, 10 x 10")
    figure.savefig(OUTPUT_DIR / "conv2_features.png", dpi=200)
    np.savez_compressed(
        OUTPUT_DIR / "feature_maps.npz",
        image=sample.numpy(),
        conv1_weights=kernels1,
        conv2_weights=kernels2,
        conv1_features=first,
        conv2_features=last,
    )
    return {
        "partition": "validation",
        "original_train_index": sample_index,
        "true_label": label,
        "predicted_label": prediction,
        "sample_rule": "first validation sample with label 7; independent of model prediction",
        "conv1_shape": list(first.shape),
        "conv2_shape": list(last.shape),
        "conv1_weight_range": [float(kernels1.min()), float(kernels1.max())],
        "conv2_weight_range": [float(kernels2.min()), float(kernels2.max())],
        "conv1_activation_max": float(first.max()),
        "conv2_activation_max": float(last.max()),
    }


torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
(OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
x_train, y_train, x_test, y_test = load_mnist()
fit_indices, validation_indices = train_test_split(np.arange(len(y_train)), test_size=0.1, stratify=y_train.numpy(), random_state=RANDOM_STATE)
x_fit, y_fit = x_train[fit_indices], y_train[fit_indices]
x_validation, y_validation = x_train[validation_indices], y_train[validation_indices]
print(f"数据划分：拟合 {len(y_fit)} / 验证 {len(y_validation)} / 测试 {len(y_test)}；设备 cpu", flush=True)
results, histories = [], {}
for config in CONFIGS:
    result, history = train(config, x_fit, y_fit, x_validation, y_validation)
    results.append(result)
    histories[config.name] = history
    write_csv(f"{config.name}_history.csv", history)
selected_names = {
    architecture: min((result for result in results if result["architecture"] == architecture), key=lambda result: validation_key(result["best"]))[
        "name"
    ]
    for architecture in ("mlp", "lenet")
}
print(f"验证集选定：{selected_names}；以下测试结果不再用于修改配置。", flush=True)
prediction_rows, report_texts, visualization = [], [], {}
for result, config in zip(results, CONFIGS, strict=True):
    model = build_model(config)
    checkpoint = torch.load(OUTPUT_DIR / "models" / f"{config.name}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    test_metrics, probabilities = evaluate(model, x_test, y_test)
    predictions = probabilities.argmax(dim=1).numpy()
    result["test"] = test_metrics
    result["confusion_matrix"] = confusion_matrix(y_test.numpy(), predictions, labels=range(10)).tolist()
    result["classification_report"] = classification_report(y_test.numpy(), predictions, labels=range(10), output_dict=True, zero_division=0)
    report_texts.append(config.name + "\n" + classification_report(y_test.numpy(), predictions, labels=range(10), digits=4, zero_division=0))
    for index, (label, prediction, probability) in enumerate(zip(y_test.tolist(), predictions.tolist(), probabilities.tolist(), strict=True)):
        prediction_rows.append(
            {
                "experiment": config.name,
                "sample_index": index,
                "true_label": label,
                "predicted_label": prediction,
                "correct": label == prediction,
                **{f"p_{digit}": value for digit, value in enumerate(probability)},
            }
        )
    if config.name == selected_names["lenet"] and isinstance(model, LeNet):
        sample_index = int(validation_indices[np.flatnonzero(y_validation.numpy() == 7)[0]])
        visualization = save_features(model, x_train[sample_index : sample_index + 1], sample_index, int(y_train[sample_index].item()))
    print(
        f"{config.name:15s} 最优轮次 {result['best']['epoch']:2d}，参数 {result['parameters']}，"
        f"测试准确率 {test_metrics['accuracy']:.4%}，损失 {test_metrics['loss']:.4f}",
        flush=True,
    )

metrics = {
    "config": {
        "random_state": RANDOM_STATE,
        "optimizer": "Adam",
        "device": "cpu",
        "threads": torch.get_num_threads(),
        "deterministic_algorithms": True,
        "preprocessing": "uint8 / 255 -> float32, NCHW, no augmentation",
        "selection_rule": "validation accuracy descending, then loss ascending; ties keep earlier epoch/config; no refit",
    },
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "packages": {name: version(name) for name in ("numpy", "torch", "scikit-learn", "matplotlib")},
    },
    "dataset": {
        "name": "MNIST",
        "source": MNIST_URL,
        "md5": dict(RESOURCES),
        "official_train_size": len(y_train),
        "fit_size": len(y_fit),
        "validation_size": len(y_validation),
        "test_size": len(y_test),
        "fit_indices": fit_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "fit_class_counts": torch.bincount(y_fit).tolist(),
        "validation_class_counts": torch.bincount(y_validation).tolist(),
        "test_class_counts": torch.bincount(y_test).tolist(),
    },
    "selected_experiments": selected_names,
    "experiments": results,
    "visualization": visualization,
    "requirement_met": all(
        next(result for result in results if result["name"] == name)["test"]["accuracy"] > 0.97 for name in selected_names.values()
    ),
}
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUTPUT_DIR / "classification_report.txt").write_text("\n".join(report_texts), encoding="utf-8")
write_csv("test_predictions.csv", prediction_rows)
write_csv("history.csv", [{"experiment": name, **row} for name, history in histories.items() for row in history])
write_csv(
    "comparison.csv",
    [
        {
            "experiment": result["name"],
            "parameters": result["parameters"],
            "epochs": result["epochs"],
            "batch_size": result["batch_size"],
            "learning_rate": result["learning_rate"],
            **result["best"],
            "test_loss": result["test"]["loss"],
            "test_accuracy": result["test"]["accuracy"],
        }
        for result in results
    ],
)
save_curves(histories, ["mlp", "mlp_deep", "lenet"], "architecture_curves.png")
save_curves(histories, ["lenet", "lenet_short", "lenet_lr_small", "lenet_batch256"], "hyperparameter_curves.png")
save_confusion_matrices(results, selected_names)
print(f"\n两类选定模型均超过 97%：{metrics['requirement_met']}；结果已保存至 {OUTPUT_DIR}", flush=True)
