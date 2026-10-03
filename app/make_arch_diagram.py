"""Render model_architecture.png — block diagram of the command classifier."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / "model_architecture.png"

STAGES = [
    ("Input audio  (16 kHz mono, 24,352 samples = 1.522 s)", "", "", "#dfe7f5", "#2b3a67"),
    ("Log-mel frontend  (NumPy — NOT in the ONNX graph)",
     "Hann(512) · rfft · power=(re²+im²)/512 · mel(64×257) · log(+1e-4)",
     "shape → (1, 1, 64, 150)", "#e3f2e1", "#2f6b3a"),
    ("stem",
     "Conv 1→64, 3×3, s1, p1 + ReLU      → (1, 64, 64, 150)",
     "640 params", "#f6e9d8", "#8a5a1c"),
    ("Stage 1",
     "Conv 64→128 + AvgPool 2×2 + 2×ResBlock(128)   → (1, 128, 32, 75)",
     "664,192 params", "#f6e9d8", "#8a5a1c"),
    ("Stage 2",
     "Conv 128→192 + AvgPool 2×2 + 2×ResBlock(192)  → (1, 192, 16, 37)",
     "1,549,248 params", "#f6e9d8", "#8a5a1c"),
    ("Stage 3",
     "Conv 192→256 + AvgPool 2×2 + 2×ResBlock(256)  → (1, 256, 8, 18)",
     "2,802,944 params", "#f6e9d8", "#8a5a1c"),
    ("Stage 4",
     "Conv 256→256 + AvgPool 2×2 + 1×ResBlock(256)  → (1, 256, 4, 9)",
     "1,770,240 params", "#f6e9d8", "#8a5a1c"),
    ("Head",
     "GlobalAvgPool → FC 256→192 + ReLU → FC 192→31   → logits (1, 31)",
     "55,327 params", "#f6e9d8", "#8a5a1c"),
    ("Decision",
     "softmax → argmax;  UNKNOWN if max(softmax) < 0.163",
     "→ 31 keyword intents  or  UNKNOWN", "#efe0f2", "#5a2f6b"),
]

fig, ax = plt.subplots(figsize=(13, 12))
n = len(STAGES)
box_h, detail_h, gap = 0.72, 0.16, 0.30
heights = [box_h + (detail_h if d else 0) for (_, d, _, _, _) in STAGES]
stack = sum(heights) + gap * (n - 1)
top, bottom = 1.35, 1.30
ylim = top + stack + bottom
ax.set_xlim(0, 10)
ax.set_ylim(0, ylim)
ax.axis("off")

# title
ax.text(5.0, ylim - 0.55, "Command Classifier — architecture",
        fontsize=17, fontweight="bold", ha="center", color="#1b2440")
ax.text(5.0, ylim - 0.95,
        "models/command_classifier/command_classifier.onnx   ·   ResNet-style 2-D CNN   ·   31 classes",
        fontsize=10, ha="center", color="#555")

y = ylim - top
for i, (title, detail, tail, fc, ec) in enumerate(STAGES):
    h = heights[i]
    y -= h
    ax.add_patch(FancyBboxPatch((0.6, y), 8.8, h,
                                boxstyle="round,pad=0.05,rounding_size=0.09",
                                linewidth=1.6, edgecolor=ec, facecolor=fc))
    ax.text(0.85, y + h - 0.24, title, fontsize=11.5, fontweight="bold", color=ec, va="top")
    if detail:
        ax.text(0.85, y + 0.18, detail, fontsize=9.2, color="#333", va="bottom")
    if tail:
        ax.text(9.25, y + 0.18, tail, fontsize=9.0, color="#111", va="bottom",
                ha="right", fontweight="bold")
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((5.0, y), (5.0, y - gap),
                                     arrowstyle="-|>", mutation_scale=16, color="#444"))
    y -= gap

# summary panel (below the stack)
ax.add_patch(FancyBboxPatch((0.6, 0.16), 8.8, 0.86,
                            boxstyle="round,pad=0.05,rounding_size=0.09",
                            linewidth=1.5, edgecolor="#1b2440", facecolor="#eef1f8"))
ax.text(5.0, 0.72,
        "TOTAL PARAMETERS: 6,842,591 (≈6.84 M)   |   27.45 MB float32   |   input (1,1,64,150) → logits (1,31)",
        fontsize=9.6, fontweight="bold", ha="center", color="#1b2440")
ax.text(5.0, 0.44,
        "opset 18 · IR 10 · PyTorch 2.14 export · CPU ONNX Runtime (2 threads)",
        fontsize=9.0, ha="center", color="#444")
ax.text(5.0, 0.26,
        "ResBlock = Conv + ReLU + Conv + Add(skip) + ReLU",
        fontsize=9.0, ha="center", color="#444")

fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)

