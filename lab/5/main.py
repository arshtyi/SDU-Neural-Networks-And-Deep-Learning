import csv
import hashlib
import json
import math
import os
import platform
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".matplotlib"))

import numpy as np
import torch
import torchtext
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

torchtext.disable_torchtext_deprecation_warning()
from torchtext.data.utils import get_tokenizer
from torchtext.datasets import AG_NEWS
from torchtext.vocab import Vocab, build_vocab_from_iterator

SEED = 42
LABELS = ("World", "Sports", "Business", "Sci/Tech")
EXAMPLE_CLASS = "Sports"
MAX_LENGTH = 128
TOKENIZE = get_tokenizer("basic_english")
EXPECTED_MD5 = {"train.csv": "b1a00f826fdfbd249f79597b59e1dc12", "test.csv": "d52ea96a97a2d943681189a97654912d"}


@dataclass(frozen=True)
class Config:
    name: str
    architecture: str
    learning_rate: float = 0.001
    embedding_dim: int = 128
    hidden_dim: int = 64
    layers: int = 1
    heads: int = 4
    dropout: float = 0.3
    batch_size: int = 128
    epochs: int = 8


CONFIGS = (
    Config("rnn", "rnn"),
    Config("rnn_lr", "rnn", learning_rate=0.003),
    Config("lstm", "lstm"),
    Config("lstm_lr", "lstm", learning_rate=0.003),
    Config("transformer", "transformer", layers=2),
    Config("transformer_lr", "transformer", learning_rate=0.0003, layers=2),
)


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def seed_everything() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


class Classifier(nn.Module):
    positions: torch.Tensor

    def __init__(self, config: Config, vocabulary_size: int) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocabulary_size, config.embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(config.dropout)
        if config.architecture == "transformer":
            position = torch.arange(MAX_LENGTH).unsqueeze(1)
            frequency = torch.exp(torch.arange(0, config.embedding_dim, 2) * (-math.log(10000) / config.embedding_dim))
            encoding = torch.zeros(MAX_LENGTH, config.embedding_dim)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency)
            self.register_buffer("positions", encoding.unsqueeze(0))
            layer = nn.TransformerEncoderLayer(
                config.embedding_dim,
                config.heads,
                config.embedding_dim * 2,
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, config.layers, enable_nested_tensor=False)
            width = config.embedding_dim
        else:
            recurrent = nn.RNN if config.architecture == "rnn" else nn.LSTM
            self.encoder = recurrent(
                config.embedding_dim,
                config.hidden_dim,
                config.layers,
                batch_first=True,
                bidirectional=True,
                dropout=config.dropout if config.layers > 1 else 0,
            )
            width = 2 * config.hidden_dim
        self.classifier = nn.Linear(width * 2, len(LABELS))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = tokens.ne(0)
        embedded = self.dropout(self.embedding(tokens))
        if isinstance(self.encoder, nn.TransformerEncoder):
            embedded = embedded * math.sqrt(self.config.embedding_dim) + self.positions[:, : tokens.shape[1]]
            encoded = self.encoder(embedded, src_key_padding_mask=~mask)
        else:
            packed = pack_padded_sequence(embedded, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
            encoded_packed, _ = self.encoder(packed)
            encoded, _ = pad_packed_sequence(encoded_packed, batch_first=True, total_length=tokens.shape[1])
        mean = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
        maximum = encoded.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(1)
        return self.classifier(self.dropout(torch.cat((mean, maximum), dim=1)))


def encode(texts: list[str], vocab: Vocab) -> torch.Tensor:
    result = torch.zeros(len(texts), MAX_LENGTH, dtype=torch.long)
    for index, text in enumerate(texts):
        ids = vocab(TOKENIZE(text)[:MAX_LENGTH]) or [1]
        result[index, : len(ids)] = torch.tensor(ids)
    return result


def prepare() -> tuple[dict, dict, Vocab]:
    data_dir = OUTPUT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    print("Loading AG_NEWS through torchtext...", flush=True)
    train_records = list(AG_NEWS(root=str(data_dir), split="train"))
    test_records = list(AG_NEWS(root=str(data_dir), split="test"))
    if len(train_records) != 120000 or len(test_records) != 7600:
        raise ValueError("Unexpected AG_NEWS split sizes")
    checksums = {path.name: hashlib.md5(path.read_bytes()).hexdigest() for path in data_dir.rglob("*.csv")}
    if checksums != EXPECTED_MD5:
        raise ValueError(f"AG_NEWS checksum mismatch: {checksums}")
    train_labels = np.array([label - 1 for label, _ in train_records])
    fit, validation = train_test_split(np.arange(120000), test_size=0.1, random_state=SEED, stratify=train_labels)
    fit_texts = [train_records[i][1] for i in fit]
    vocab = build_vocab_from_iterator((TOKENIZE(text) for text in fit_texts), min_freq=2, specials=["<pad>", "<unk>"], max_tokens=30000)
    vocab.set_default_index(1)
    save_json(OUTPUT / "vocab.json", vocab.get_itos())
    np.savez_compressed(data_dir / "split_indices.npz", fit=fit, validation=validation)
    tensor_data = {}
    statistics = {}
    for split, texts, labels in (
        ("fit", fit_texts, train_labels[fit]),
        ("validation", [train_records[i][1] for i in validation], train_labels[validation]),
        ("test", [text for _, text in test_records], np.array([label - 1 for label, _ in test_records])),
    ):
        lengths = np.array([len(TOKENIZE(text)) for text in texts])
        tensor_data[split] = (encode(texts, vocab), torch.tensor(labels, dtype=torch.long))
        statistics[split] = {
            "size": len(texts),
            "class_counts": np.bincount(labels, minlength=4).tolist(),
            "mean_tokens": float(lengths.mean()),
            "median_tokens": float(np.median(lengths)),
            "p95_tokens": float(np.percentile(lengths, 95)),
            "max_tokens": int(lengths.max()),
            "truncated_count": int((lengths > MAX_LENGTH).sum()),
        }
    metadata = {
        "name": "AG_NEWS",
        "loader": "torchtext.datasets.AG_NEWS",
        "official_train_size": 120000,
        "labels": LABELS,
        "label_mapping": "original labels 1..4 -> model labels 0..3",
        "seed": SEED,
        "vocabulary_size": len(vocab),
        "vocabulary_source": "fit split only",
        "min_frequency": 2,
        "max_length": MAX_LENGTH,
        "tokenizer": "torchtext basic_english",
        "splits": statistics,
        "md5": checksums,
    }
    save_json(data_dir / "metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return tensor_data, metadata, vocab


def batches(x: torch.Tensor, y: torch.Tensor, size: int, shuffle: bool = False, generator: torch.Generator | None = None) -> Iterator:
    order = torch.randperm(len(y), generator=generator) if shuffle else torch.arange(len(y))
    lengths = x.ne(0).sum(1)
    chunks = []
    for pool in order.split(size * 20):
        chunks.extend(pool[lengths[pool].argsort(descending=True)].split(size))
    if shuffle:
        chunks = [chunks[i] for i in torch.randperm(len(chunks), generator=generator)]
    for indices in chunks:
        yield indices, x[indices, : int(lengths[indices].max())], y[indices]


@torch.inference_mode()
def evaluate(model: Classifier, x: torch.Tensor, y: torch.Tensor) -> tuple[dict, torch.Tensor]:
    model.eval()
    probabilities = torch.zeros(len(y), 4)
    loss_sum = 0.0
    start = perf_counter()
    for indices, tokens, labels in batches(x, y, 256):
        logits = model(tokens)
        loss_sum += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        probabilities[indices] = logits.softmax(1)
    correct = int((probabilities.argmax(1) == y).sum())
    return {"loss": loss_sum / len(y), "accuracy": correct / len(y), "correct": correct, "seconds": perf_counter() - start}, probabilities


def validation_key(row: dict) -> tuple[float, float]:
    return -row["validation_accuracy"], row["validation_loss"]


def train(config: Config, data: dict, metadata: dict) -> dict:
    seed_everything()
    generator = torch.Generator().manual_seed(SEED)
    model = Classifier(config, metadata["vocabulary_size"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.learning_rate * 0.1)
    x, y = data["fit"]
    history = []
    best = None
    start = perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = correct = count = 0
        epoch_start = perf_counter()
        for _, tokens, labels in batches(x, y, config.batch_size, True, generator):
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = nn.functional.cross_entropy(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss: {config.name}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            loss_sum += loss.item() * len(labels)
            correct += int((logits.argmax(1) == labels).sum().item())
            count += len(labels)
        training_seconds = perf_counter() - epoch_start
        validation, _ = evaluate(model, *data["validation"])
        row = {
            "epoch": epoch,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train_loss": loss_sum / count,
            "train_accuracy": correct / count,
            "validation_loss": validation["loss"],
            "validation_accuracy": validation["accuracy"],
            "training_seconds": training_seconds,
            "validation_seconds": validation["seconds"],
        }
        history.append(row)
        if best is None or validation_key(row) < validation_key(best):
            best = row.copy()
            checkpoint = {
                "config": asdict(config),
                "vocabulary_size": metadata["vocabulary_size"],
                "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
                "best": best,
            }
            torch.save(checkpoint, OUTPUT / "models" / f"{config.name}.pt")
        scheduler.step()
        print(
            f"{config.name:16s} {epoch:02d}/{config.epochs} train loss={row['train_loss']:.4f} acc={row['train_accuracy']:.2%} "
            f"val loss={row['validation_loss']:.4f} acc={row['validation_accuracy']:.2%} train={training_seconds:.1f}s",
            flush=True,
        )
    result = {
        "name": config.name,
        "config": asdict(config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "best": best,
        "last": history[-1],
        "history": history,
        "training_seconds": sum(row["training_seconds"] for row in history),
        "wall_seconds": perf_counter() - start,
        "device": "cpu",
    }
    return result


def restore(name: str) -> Classifier:
    checkpoint = torch.load(OUTPUT / "models" / f"{name}.pt", map_location="cpu", weights_only=True)
    model = Classifier(Config(**checkpoint["config"]), checkpoint["vocabulary_size"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


@torch.inference_mode()
def predict_text(model: Classifier, text: str, vocab: Vocab) -> dict:
    tokens = TOKENIZE(text)
    if not tokens:
        raise ValueError("Input text must contain at least one token")
    model.eval()
    encoded = vocab(tokens)
    chunks = [encoded[start : start + MAX_LENGTH] for start in range(0, len(encoded), MAX_LENGTH)]
    x = torch.zeros(len(chunks), min(MAX_LENGTH, len(encoded)), dtype=torch.long)
    for i, chunk in enumerate(chunks):
        x[i, : len(chunk)] = torch.tensor(chunk)
    probabilities = torch.cat([model(batch).softmax(1) for batch in x.split(64)])
    weights = torch.tensor([len(chunk) for chunk in chunks], dtype=torch.float32)
    combined = (probabilities * weights.unsqueeze(1)).sum(0) / weights.sum()
    predicted = int(combined.argmax())
    return {
        "token_count": len(tokens),
        "chunk_count": len(chunks),
        "chunk_lengths": [len(chunk) for chunk in chunks],
        "unknown_tokens": encoded.count(1),
        "predicted_label": predicted,
        "predicted_class": LABELS[predicted],
        "probabilities": dict(zip(LABELS, combined.tolist(), strict=True)),
        "confidence": float(combined[predicted]),
    }


def verify_models() -> dict:
    seed_everything()
    checks = []
    vocabulary = build_vocab_from_iterator([["news", "sports"]], specials=["<pad>", "<unk>"])
    vocabulary.set_default_index(1)
    for config in CONFIGS[::2]:
        model = Classifier(config, 20).eval()
        tokens = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])
        with torch.no_grad():
            logits = model(tokens)
            torch.testing.assert_close(logits, model(nn.functional.pad(tokens, (0, 7))), rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(logits[:1], model(tokens[:1, :3]), rtol=1e-4, atol=1e-5)
        assert logits.shape == (2, 4) and torch.isfinite(logits).all()
        loss = nn.functional.cross_entropy(model(tokens), torch.tensor([0, 1]))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert model.embedding.weight.grad is not None and model.embedding.weight.grad[0].count_nonzero() == 0
        restored = Classifier(config, 20).eval()
        restored.load_state_dict(model.state_dict())
        with torch.no_grad():
            torch.testing.assert_close(logits, restored(tokens), rtol=0, atol=0)
        long_prediction = predict_text(model, "news " * (2 * MAX_LENGTH + 3), vocabulary)
        assert long_prediction["chunk_lengths"] == [128, 128, 3]
        assert long_prediction["token_count"] == sum(long_prediction["chunk_lengths"])
        assert math.isclose(sum(long_prediction["probabilities"].values()), 1.0, abs_tol=1e-6)
        unknown_prediction = predict_text(model, "unseenword", vocabulary)
        assert unknown_prediction["unknown_tokens"] == 1
        try:
            predict_text(model, "   ", vocabulary)
        except ValueError:
            pass
        else:
            raise AssertionError("Empty input must be rejected")
        checks.append(config.architecture)
    return {
        "passed": True,
        "architectures": checks,
        "checks": [
            "padding_invariance",
            "batch_invariance",
            "finite_gradients",
            "zero_pad_gradient",
            "state_restore",
            "long_text_coverage",
            "probability_normalization",
            "unknown_token",
            "empty_text_rejection",
        ],
    }


def plots(results: list[dict], selected: dict) -> None:
    figure = Figure(figsize=(12, 8), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3)
    for column, architecture in enumerate(("rnn", "lstm", "transformer")):
        for result in results:
            if result["config"]["architecture"] != architecture:
                continue
            history = result["history"]
            epochs = [row["epoch"] for row in history]
            label = f"lr={result['config']['learning_rate']:g}"
            line = axes[0, column].plot(epochs, [row["validation_accuracy"] * 100 for row in history], label=label)[0]
            axes[0, column].plot(epochs, [row["train_accuracy"] * 100 for row in history], "--", color=line.get_color(), alpha=0.6)
            axes[1, column].plot(epochs, [row["validation_loss"] for row in history], color=line.get_color(), label=label)
            axes[1, column].plot(epochs, [row["train_loss"] for row in history], "--", color=line.get_color(), alpha=0.6)
        axes[0, column].set_title(architecture.upper())
        axes[0, column].set_ylabel("Accuracy (%)")
        axes[1, column].set_ylabel("Cross-entropy loss")
        for axis in axes[:, column]:
            axis.set_xlabel("Epoch")
            axis.grid(alpha=0.2)
            axis.legend()
    figure.suptitle("Solid: validation (eval); dashed: training (online, dropout enabled)")
    figure.savefig(OUTPUT / "learning_curves.png", dpi=180)
    figure = Figure(figsize=(12, 3.8), layout="constrained")
    FigureCanvasAgg(figure)
    for axis, (architecture, name) in zip(figure.subplots(1, 3), selected.items(), strict=True):
        result = next(item for item in results if item["name"] == name)
        matrix = np.array(result["confusion_matrix"])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1900)
        for row in range(4):
            for column in range(4):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > 950 else "black")
        axis.set(
            xticks=range(4), yticks=range(4), xticklabels=LABELS, yticklabels=LABELS, xlabel="Predicted", ylabel="True", title=architecture.upper()
        )
        axis.tick_params(axis="x", rotation=30)
    figure.savefig(OUTPUT / "confusion_matrices.png", dpi=180)


def summarize(results: list[dict], data: dict, metadata: dict, vocab: Vocab) -> None:
    selected = {
        architecture: min((r for r in results if r["config"]["architecture"] == architecture), key=lambda r: validation_key(r["best"]))["name"]
        for architecture in ("rnn", "lstm", "transformer")
    }
    example = (ROOT / "example.txt").read_text(encoding="utf-8").strip()
    print(f"Long text (true class: {EXAMPLE_CLASS}):\n{example}", flush=True)
    predictions = []
    for result in results:
        model = restore(result["name"])
        test, probabilities = evaluate(model, *data["test"])
        labels = data["test"][1].numpy()
        predicted = probabilities.argmax(1).numpy()
        result["test"] = test
        result["confusion_matrix"] = confusion_matrix(labels, predicted, labels=range(4)).tolist()
        result["top_confusions"] = sorted(
            ({"true": LABELS[i], "predicted": LABELS[j], "count": result["confusion_matrix"][i][j]} for i in range(4) for j in range(4) if i != j),
            key=lambda item: -int(item["count"]),
        )[:3]
        result["classification_report"] = classification_report(
            labels, predicted, labels=range(4), target_names=LABELS, output_dict=True, zero_division=0
        )
        np.savez_compressed(OUTPUT / "runs" / f"{result['name']}_test.npz", labels=labels, probabilities=probabilities.numpy())
        if result["name"] in selected.values():
            prediction = {"model": result["name"], **predict_text(model, example, vocab)}
            prediction["correct"] = prediction["predicted_class"] == EXAMPLE_CLASS
            predictions.append(prediction)
            print(
                f"{result['name']}: test={test['accuracy']:.2%}, long text true={EXAMPLE_CLASS}, predicted={prediction['predicted_class']}",
                flush=True,
            )
    metrics = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "device": "cpu",
            "threads": torch.get_num_threads(),
            "run_at": datetime.now().astimezone().isoformat(),
            "packages": {name: version(name) for name in ("torch", "torchtext", "torchdata", "numpy", "scikit-learn", "matplotlib", "ty", "ruff")},
        },
        "config": {
            "seed": SEED,
            "optimizer": "AdamW",
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "scheduler": "cosine decay to 0.1 * initial learning rate",
            "selection_rule": "validation accuracy descending, loss ascending; earlier on ties",
            "training_metrics": "sample-weighted online metrics with dropout; validation/test use eval",
            "pretrained_embeddings": False,
        },
        "dataset": metadata,
        "verification": verify_models(),
        "experiments": results,
        "selected_experiments": selected,
        "long_text": {
            "text": example,
            "true_class": EXAMPLE_CLASS,
            "aggregation": "non-overlapping 128-token chunks; length-weighted mean of softmax probabilities",
            "predictions": predictions,
        },
        "requirement_met": all(r["test"]["accuracy"] >= 0.9 for r in results if r["name"] in selected.values()),
    }
    save_json(OUTPUT / "metrics.json", metrics)
    save_json(OUTPUT / "long_text_predictions.json", metrics["long_text"])
    with (OUTPUT / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["experiment", *results[0]["history"][0]])
        writer.writeheader()
        for result in results:
            writer.writerows({"experiment": result["name"], **row} for row in result["history"])
    plots(results, selected)
    print("All three selected models reach 90%:", metrics["requirement_met"], flush=True)


torch.set_num_threads(2)
for directory in (OUTPUT, OUTPUT / "models", OUTPUT / "runs"):
    directory.mkdir(parents=True, exist_ok=True)
data, metadata, vocab = prepare()
results = [train(config, data, metadata) for config in CONFIGS]
summarize(results, data, metadata, vocab)
