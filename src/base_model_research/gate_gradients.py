"""Pre-clipping weight gradient RMS for the residual-tanh proxy experiment."""

import csv
import json
import math

import torch


class GateGradientLog:
    def __init__(self, stream, model):
        self.stream = stream
        self.writer = csv.writer(stream)
        self.projections = [
            projection.weight
            for layer in model.layers
            for gate in (layer.attention_gate, layer.ffn_gate)
            for projection in (gate.current, gate.previous)
        ]
        self.writer.writerow(
            ["update", "layer", "attention_current", "attention_previous", "ffn_current", "ffn_previous"]
        )
        self.stream.flush()

    def write(self, update):
        # All four gate matrices have the same shape. Reduce in FP64 so tiny or
        # large finite FP32 gradients survive squaring, then transfer once.
        grads = [p.grad for p in self.projections if p.grad is not None]
        reduced = []
        if grads:
            batch = torch.stack(grads).flatten(1)
            norms = torch.linalg.vector_norm(batch, dim=1, dtype=torch.float64)
            reduced = (norms / math.sqrt(batch.shape[1])).tolist()
        available = iter(reduced)
        rms = [next(available) if p.grad is not None else None for p in self.projections]
        for start in range(0, len(rms), 4):
            self.writer.writerow([update, start // 4 + 1, *rms[start : start + 4]])
        self.stream.flush()
        if update in (1, 8, 16, 32) or update % 50 == 0:
            layers = sorted({0, (len(rms) // 4 - 1) // 2, len(rms) // 4 - 1})
            for i in layers:
                values = "  ".join("missing" if v is None else f"{v:.2e}" for v in rms[4 * i : 4 * i + 4])
                print(
                    f"Update {update} · layer {i + 1} · gradient RMS [attn W1, W2, FFN W1, W2]  {values}",
                    flush=True,
                )


def plot_gate_training(run_dir):
    """Plot the recorded gradients and validation losses, including partial runs."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.colors import Normalize
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    path = run_dir / "gate_gradients.csv"
    if not path.exists():
        print("The trainer has not started logging gate gradients yet.")
        return
    data = pd.read_csv(path)
    metrics = run_dir / "metrics.jsonl"
    losses = []
    if metrics.exists():
        with metrics.open() as stream:
            for line in stream:
                row = json.loads(line)
                if "validation_loss" in row:
                    losses.append((row["step"], row["validation_loss"]))

    fig = plt.figure(figsize=(12, 9), layout="constrained")
    grid = fig.add_gridspec(3, 2)
    axes = [fig.add_subplot(grid[i // 2, i % 2]) for i in range(4)]
    validation = fig.add_subplot(grid[2, :])
    projections = ("attention_current", "attention_previous", "ffn_current", "ffn_previous")
    if data.empty:
        for axis in axes:
            axis.set_visible(False)
        print("No update reached the end of gradient accumulation.")
    else:
        values = data[list(projections)].to_numpy()
        positive = values[np.isfinite(values) & (values > 0)]
        low = int(np.floor(np.log10(positive.min()))) if positive.size else -1
        high = max(low + 1, int(np.ceil(np.log10(positive.max())))) if positive.size else 0
        scale = Normalize(vmin=low, vmax=high)
        colors = plt.get_cmap("viridis").with_extremes(under="#dddddd", bad="#ce3b31")
        layers = range(1, json.loads((run_dir / "run.json").read_text())["model"]["n_layers"] + 1)
        updates = range(1, int(data["update"].max()) + 1)
        for axis, column, title in zip(
            axes, projections, ("Attention W1", "Attention W2", "FFN W1", "FFN W2")
        ):
            table = data.pivot(index="layer", columns="update", values=column).reindex(
                index=layers, columns=updates
            )
            raw = table.to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                logged = np.where(raw == 0, low - 1, np.log10(raw))
            image = axis.imshow(
                np.ma.masked_invalid(logged),
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                cmap=colors,
                norm=scale,
                extent=(0.5, len(updates) + 0.5, 0.5, len(layers) + 0.5),
            )
            axis.set(title=title, xlabel="Optimizer update", ylabel="Layer")
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.set_yticks(np.unique(np.linspace(1, len(layers), min(5, len(layers))).round().astype(int)))
        fig.colorbar(image, ax=axes, label="log₁₀(weight gradient RMS)")
        axes[0].legend(
            handles=[
                Patch(color="#dddddd", label="Exactly zero"),
                Patch(color="#ce3b31", label="Missing or nonfinite"),
            ],
            loc="lower left",
            bbox_to_anchor=(0, 1.12),
            ncol=2,
            frameon=False,
        )

    if losses:
        steps, values = zip(*losses)
        validation.plot(steps, values, marker="o", color="#156479")
        print(f"Latest validation loss {values[-1]:.4f} at update {steps[-1]:,}.")
    else:
        validation.set(xticks=[], yticks=[])
        validation.text(
            0.5, 0.5, "No validation completed yet", transform=validation.transAxes, ha="center", va="center"
        )
    validation.set(title="Held-out FineWeb-Edu", xlabel="Optimizer update", ylabel="Validation loss")
    validation.grid(axis="y", alpha=0.25)
    output = run_dir / "gate_gradients_and_validation.png"
    fig.savefig(output, dpi=140)
    print(f"Saved plot to {output}")
    return fig
