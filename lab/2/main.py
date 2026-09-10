import csv
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import ArrayLike, NDArray
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS = 5
K_VALUES = tuple(range(1, 20, 2))
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


class KNNClassifier:
    def __init__(self, n_neighbors: int = 3) -> None:
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError("n_neighbors 必须是正整数。")
        self.n_neighbors = n_neighbors
        self._x_train: NDArray[np.float64] | None = None
        self._classes: NDArray[np.int64] | None = None
        self._encoded_y: NDArray[np.intp] | None = None

    def fit(self, x: ArrayLike, y: ArrayLike) -> "KNNClassifier":
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y)
        if x_array.ndim != 2 or 0 in x_array.shape:
            raise ValueError("训练特征必须是非空二维数组。")
        if not np.isfinite(x_array).all():
            raise ValueError("训练特征不能包含 NaN 或无穷值。")
        if y_array.ndim != 1 or len(y_array) != len(x_array):
            raise ValueError("标签必须是一维数组，且与训练样本数量一致。")
        if not np.issubdtype(y_array.dtype, np.integer):
            raise ValueError("类别标签必须为整数。")
        if self.n_neighbors > len(x_array):
            raise ValueError("K 不能大于训练样本数量。")
        self._x_train = x_array.copy()
        self._classes, self._encoded_y = np.unique(y_array.astype(np.int64), return_inverse=True)
        return self

    def predict(self, x: ArrayLike) -> NDArray[np.int64]:
        if self._x_train is None or self._classes is None or self._encoded_y is None:
            raise RuntimeError("请先调用 fit，再调用 predict。")
        x_array = np.asarray(x, dtype=np.float64)
        if x_array.ndim != 2 or x_array.shape[1] != self._x_train.shape[1]:
            raise ValueError("预测特征必须是二维数组，且特征数与训练集相同。")
        if not np.isfinite(x_array).all():
            raise ValueError("预测特征不能包含 NaN 或无穷值。")
        predictions = np.empty(len(x_array), dtype=np.int64)
        for index, sample in enumerate(x_array):
            squared_distances = np.sum((self._x_train - sample) ** 2, axis=1)
            neighbors = np.argsort(squared_distances, kind="stable")[: self.n_neighbors]
            votes = np.bincount(self._encoded_y[neighbors], minlength=len(self._classes))
            predictions[index] = self._classes[np.argmax(votes)]
        return predictions


@dataclass(frozen=True)
class CVResult:
    k: int
    fold_scores: tuple[float, ...]

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.fold_scores))

    @property
    def std_accuracy(self) -> float:
        return float(np.std(self.fold_scores, ddof=0))

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "fold_scores": list(self.fold_scores),
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
        }


def cross_validate_knn(x_train: NDArray, y_train: NDArray) -> tuple[CVResult, list[CVResult]]:
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    folds = list(splitter.split(x_train, y_train))
    results = []
    for k in K_VALUES:
        scores = []
        for fit_indices, validation_indices in folds:
            scaler = StandardScaler()
            x_fit = scaler.fit_transform(x_train[fit_indices])
            x_validation = scaler.transform(x_train[validation_indices])
            model = KNNClassifier(n_neighbors=k).fit(x_fit, y_train[fit_indices])
            predictions = model.predict(x_validation)
            scores.append(float(accuracy_score(y_train[validation_indices], predictions)))
        results.append(CVResult(k=k, fold_scores=tuple(scores)))
    best = min(results, key=lambda result: (-result.mean_accuracy, result.k))
    return best, results


def save_plots(results: list[CVResult], best_k: int, matrix: NDArray, class_names: list[str]) -> None:
    figure = Figure(figsize=(7, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    axis.errorbar(
        [result.k for result in results],
        [result.mean_accuracy for result in results],
        yerr=[result.std_accuracy for result in results],
        fmt="o-",
        capsize=4,
        label="Validation mean +/- 1 std",
    )
    axis.axvline(best_k, color="tab:orange", linestyle="--", label=f"Selected K = {best_k}")
    axis.set(xlabel="Number of neighbors (K)", ylabel="Accuracy", title="Stratified 5-fold cross-validation")
    axis.set_xticks(K_VALUES)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(OUTPUT_DIR / "cv_accuracy.png", dpi=180)
    figure = Figure(figsize=(6, 5), layout="constrained")
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    heatmap = axis.imshow(matrix, cmap="Blues", vmin=0)
    figure.colorbar(heatmap, ax=axis)
    axis.set_xticks(range(len(class_names)), labels=class_names)
    axis.set_yticks(range(len(class_names)), labels=class_names)
    axis.set(xlabel="Predicted label", ylabel="True label", title="Test-set confusion matrix")
    for row in range(len(class_names)):
        for column in range(len(class_names)):
            color = "white" if matrix[row, column] > matrix.max() / 2 else "black"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)


def main() -> None:
    iris = load_iris()
    x = np.asarray(iris.data, dtype=np.float64)
    y = np.asarray(iris.target, dtype=np.int64)
    class_names = [str(name) for name in iris.target_names]
    labels = np.arange(len(class_names))
    train_indices, test_indices = train_test_split(np.arange(len(y)), test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    x_train, x_test = x[train_indices], x[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    best, cv_results = cross_validate_knn(x_train, y_train)
    print(f"数据集：{len(y)} 个样本，{x.shape[1]} 个特征，{len(class_names)} 个类别")
    print(f"训练集：{len(y_train)}；测试集：{len(y_test)}；随机种子：{RANDOM_STATE}")
    print("\nK 值与五折验证结果（均值和标准差）：")
    for result in cv_results:
        fold_text = ", ".join(f"{score:.4f}" for score in result.fold_scores)
        print(f"K={result.k:2d}: [{fold_text}]  {result.mean_accuracy:.4f} +/- {result.std_accuracy:.4f}")
    print(f"\n最优 K：{best.k}，交叉验证平均准确率：{best.mean_accuracy:.4f}")
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    final_model = KNNClassifier(n_neighbors=best.k).fit(x_train_scaled, y_train)
    predictions = final_model.predict(x_test_scaled)
    test_accuracy = float(accuracy_score(y_test, predictions))
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    report = classification_report(y_test, predictions, labels=labels, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, predictions, labels=labels, target_names=class_names, digits=4, zero_division=0)
    print(f"测试集准确率：{test_accuracy:.4f}（{np.count_nonzero(predictions == y_test)}/{len(y_test)}）")
    print("\n分类报告：\n" + report_text)
    print("混淆矩阵（行：真实类别；列：预测类别）：\n", matrix)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / "cv_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["k", *[f"fold_{index}" for index in range(1, N_SPLITS + 1)], "mean_accuracy", "std_accuracy"])
        for result in cv_results:
            writer.writerow([result.k, *result.fold_scores, result.mean_accuracy, result.std_accuracy])
    with (OUTPUT_DIR / "test_predictions.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", *iris.feature_names, "true_label", "predicted_label", "true_name", "predicted_name", "correct"])
        for index, true_label, predicted_label in zip(test_indices, y_test, predictions, strict=True):
            writer.writerow(
                [
                    index,
                    *x[index],
                    true_label,
                    predicted_label,
                    class_names[true_label],
                    class_names[predicted_label],
                    bool(true_label == predicted_label),
                ]
            )
    metrics = {
        "config": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "n_splits": N_SPLITS,
            "k_values": list(K_VALUES),
            "distance": "euclidean",
            "standardization": "StandardScaler fitted only on each training partition",
            "vote_tie_break": "smallest class label",
            "selection_tie_break": "smallest k",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {name: version(name) for name in ("numpy", "scikit-learn", "matplotlib")},
        },
        "dataset": {
            "n_samples": len(y),
            "n_features": x.shape[1],
            "class_names": class_names,
            "train_size": len(y_train),
            "test_size": len(y_test),
            "train_class_counts": np.bincount(y_train, minlength=len(class_names)).tolist(),
            "test_class_counts": np.bincount(y_test, minlength=len(class_names)).tolist(),
            "train_indices": train_indices.tolist(),
            "test_indices": test_indices.tolist(),
        },
        "best_k": best.k,
        "cv_results": [result.as_dict() for result in cv_results],
        "best_cv_mean_accuracy": best.mean_accuracy,
        "best_cv_std_accuracy": best.std_accuracy,
        "test_accuracy": test_accuracy,
        "test_correct": int(np.count_nonzero(predictions == y_test)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    save_plots(cv_results, best.k, matrix, class_names)
    print(f"\n指标、逐样本预测及图表已保存至：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
