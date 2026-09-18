import csv
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn

RANDOM_STATE = 42
EPOCHS = 200
BATCH_SIZE = 16
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


@dataclass(frozen=True)
class Config:
    name: str
    preprocessing: str = "standard"
    hidden_sizes: tuple[int, ...] = (16,)
    learning_rate: float = 0.01


CONFIGS = (
    Config("raw", preprocessing="raw"),
    Config("standard"),
    Config("minmax", preprocessing="minmax"),
    Config("wide", hidden_sizes=(32,)),
    Config("deep", hidden_sizes=(16, 8)),
    Config("lr_small", learning_rate=0.001),
    Config("lr_large", learning_rate=0.05),
)


def preprocess(x_train: np.ndarray, x_eval: np.ndarray, method: str) -> tuple[np.ndarray, np.ndarray, dict]:
    if method == "raw":
        return x_train.astype(np.float32), x_eval.astype(np.float32), {"method": method}
    scaler = StandardScaler() if method == "standard" else MinMaxScaler()
    train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    eval_scaled = scaler.transform(x_eval).astype(np.float32)
    parameters = {"method": method, "scale": np.asarray(scaler.scale_).tolist()}
    if isinstance(scaler, StandardScaler):
        parameters["mean"] = np.asarray(scaler.mean_).tolist()
    else:
        parameters["min"] = scaler.min_.tolist()
    return train_scaled, eval_scaled, parameters


def build_model(hidden_sizes: tuple[int, ...]) -> nn.Sequential:
    layers = []
    in_features = 4
    for width in hidden_sizes:
        layers.extend((nn.Linear(in_features, width), nn.ReLU()))
        in_features = width
    layers.append(nn.Linear(in_features, 3))
    return nn.Sequential(*layers)


def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(x)
        return {
            "loss": nn.functional.cross_entropy(logits, y).item(),
            "accuracy": (logits.argmax(dim=1) == y).float().mean().item(),
        }


def train(
    config: Config, x_train: np.ndarray, y_train: np.ndarray, validation: tuple[np.ndarray, np.ndarray] | None = None
) -> tuple[nn.Sequential, list[dict]]:
    torch.manual_seed(RANDOM_STATE)
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    model = build_model(config.hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    x, y = torch.from_numpy(x_train), torch.from_numpy(y_train)
    validation_tensors = None if validation is None else tuple(torch.from_numpy(array) for array in validation)
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for indices in torch.randperm(len(y), generator=generator).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss = criterion(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in evaluate(model, x, y).items()}}
        if validation_tensors is not None:
            row.update({f"validation_{key}": value for key, value in evaluate(model, *validation_tensors).items()})
        history.append(row)
    return model, history


def save_curves(histories: dict[str, list[dict]], names: list[str], filename: str) -> None:
    figure = Figure(figsize=(10, 6), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    for row, partition in enumerate(("train", "validation")):
        for column, metric in enumerate(("loss", "accuracy")):
            axis = axes[row, column]
            for name in names:
                history = histories[name]
                axis.plot([item["epoch"] for item in history], [item[f"{partition}_{metric}"] for item in history], label=name)
            axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"{partition.capitalize()} {metric}")
            if metric == "accuracy":
                axis.set_ylim(0, 1.03)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
    figure.savefig(OUTPUT_DIR / filename, dpi=180)


def save_confusion_matrix(matrix: np.ndarray, class_names: list[str]) -> None:
    figure = Figure(figsize=(5, 4), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.imshow(matrix, cmap="Blues", vmin=0)
    axis.set_xticks(range(3), labels=class_names)
    axis.set_yticks(range(3), labels=class_names)
    axis.set(xlabel="Predicted label", ylabel="True label", title="Selected model: test confusion matrix")
    for row in range(3):
        for column in range(3):
            axis.text(
                column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() / 2 else "black"
            )
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)


def write_csv(filename: str, rows: list[dict]) -> None:
    with (OUTPUT_DIR / filename).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
iris = load_iris()
x = np.asarray(iris.data, dtype=np.float64)
y = np.asarray(iris.target, dtype=np.int64)
class_names = [str(name) for name in iris.target_names]
train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=RANDOM_STATE)
fit_indices, validation_indices = train_test_split(train_indices, test_size=0.2, stratify=y[train_indices], random_state=RANDOM_STATE)
(OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)

histories, selection_results = {}, {}
print(f"数据划分：训练 {len(train_indices)}（拟合 {len(fit_indices)} / 验证 {len(validation_indices)}），测试 {len(test_indices)}")
for config in CONFIGS:
    x_fit, x_validation, _ = preprocess(x[fit_indices], x[validation_indices], config.preprocessing)
    _, history = train(config, x_fit, y[fit_indices], (x_validation, y[validation_indices]))
    histories[config.name] = history
    selection_results[config.name] = history[-1]
    print(f"{config.name:10s} 验证准确率 {history[-1]['validation_accuracy']:.4f}，验证损失 {history[-1]['validation_loss']:.4f}")

selected = min(
    CONFIGS, key=lambda config: (-selection_results[config.name]["validation_accuracy"], selection_results[config.name]["validation_loss"])
)
print(f"验证集选定配置：{selected.name}\n随后从头在完整训练集上训练各配置，测试结果仅用于固定方案的对照。")
results, predictions_rows, history_rows, report_texts = [], [], [], []
for config in CONFIGS:
    x_train, x_test, scaling = preprocess(x[train_indices], x[test_indices], config.preprocessing)
    model, refit_history = train(config, x_train, y[train_indices])
    test_metrics = evaluate(model, torch.from_numpy(x_test), torch.from_numpy(y[test_indices]))
    with torch.no_grad():
        probabilities = model(torch.from_numpy(x_test)).softmax(dim=1).numpy()
    predictions = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y[test_indices], predictions, labels=[0, 1, 2])
    report = classification_report(y[test_indices], predictions, labels=[0, 1, 2], target_names=class_names, output_dict=True, zero_division=0)
    results.append(
        {
            **asdict(config),
            "selection": selection_results[config.name],
            "final_train": refit_history[-1],
            "test": {**test_metrics, "correct": int(np.count_nonzero(predictions == y[test_indices]))},
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
        }
    )
    torch.save(
        {"config": asdict(config), "state_dict": model.state_dict(), "preprocessing": scaling, "class_names": class_names},
        OUTPUT_DIR / "models" / f"{config.name}.pt",
    )
    for index, prediction, probability in zip(test_indices, predictions, probabilities, strict=True):
        predictions_rows.append(
            {
                "experiment": config.name,
                "sample_index": int(index),
                **dict(zip(iris.feature_names, x[index].tolist(), strict=True)),
                "true_label": int(y[index]),
                "predicted_label": int(prediction),
                "true_name": class_names[y[index]],
                "predicted_name": class_names[prediction],
                "correct": bool(y[index] == prediction),
                **{f"p_{name}": float(value) for name, value in zip(class_names, probability, strict=True)},
            }
        )
    for stage, history in (("selection", histories[config.name]), ("refit", refit_history)):
        history_rows.extend(
            {
                "experiment": config.name,
                "stage": stage,
                **row,
                "validation_loss": row.get("validation_loss", ""),
                "validation_accuracy": row.get("validation_accuracy", ""),
            }
            for row in history
        )
    report_texts.append(f"{config.name}\n" + classification_report(y[test_indices], predictions, target_names=class_names, digits=4, zero_division=0))
    print(f"{config.name:10s} 测试准确率 {test_metrics['accuracy']:.4f}（{results[-1]['test']['correct']}/30），测试损失 {test_metrics['loss']:.4f}")
    if config == selected:
        save_confusion_matrix(matrix, class_names)

metrics = {
    "config": {
        "random_state": RANDOM_STATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": "Adam",
        "device": "cpu",
        "selection_rule": "final validation accuracy descending, then loss ascending, then config order",
    },
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("numpy", "torch", "scikit-learn", "matplotlib")},
    },
    "dataset": {
        "class_names": class_names,
        "train_indices": train_indices.tolist(),
        "fit_indices": fit_indices.tolist(),
        "validation_indices": validation_indices.tolist(),
        "test_indices": test_indices.tolist(),
    },
    "selected_experiment": selected.name,
    "experiments": results,
}
(OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
(OUTPUT_DIR / "classification_report.txt").write_text("\n".join(report_texts), encoding="utf-8")
write_csv("history.csv", history_rows)
write_csv("test_predictions.csv", predictions_rows)
write_csv(
    "comparison.csv",
    [
        {
            "experiment": result["name"],
            "hidden_sizes": "-".join(map(str, result["hidden_sizes"])),
            "learning_rate": result["learning_rate"],
            "validation_accuracy": result["selection"]["validation_accuracy"],
            "validation_loss": result["selection"]["validation_loss"],
            "train_accuracy": result["final_train"]["train_accuracy"],
            "test_accuracy": result["test"]["accuracy"],
            "test_loss": result["test"]["loss"],
        }
        for result in results
    ],
)
save_curves(histories, ["raw", "standard", "minmax"], "preprocessing_curves.png")
save_curves(histories, ["standard", "wide", "deep", "lr_small", "lr_large"], "hyperparameter_curves.png")
print(f"\n模型、指标、逐轮记录及图表已保存至：{OUTPUT_DIR}")
