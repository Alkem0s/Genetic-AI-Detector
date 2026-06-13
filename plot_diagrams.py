# plot_diagrams.py
"""
Clean, paper-ready diagram generator.
Run:  python plot_diagrams.py
Outputs: paper/general_pipeline.png, paper/model_architecture.png
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def rbox(ax, cx, cy, w, h, fc, ec, lw=1.4, radius=0.12):
    """Rounded rectangle centred at (cx, cy)."""
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        clip_on=False, zorder=3
    )
    ax.add_patch(patch)
    return patch


def label(ax, cx, cy, text, fs=9, color='#1a1a2e', bold=False, va='center'):
    fw = 'bold' if bold else 'normal'
    ax.text(cx, cy, text, ha='center', va=va, fontsize=fs,
            color=color, fontweight=fw, zorder=4,
            multialignment='center')


def arrow(ax, x1, y1, x2, y2, color='#37474f', lw=1.3, style='->',
          ls='-', connectionstyle='arc3,rad=0.0'):
    ax.annotate(
        '', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle=style, color=color, lw=lw,
            linestyle=ls,
            connectionstyle=connectionstyle,
        ),
        zorder=5
    )


def circle_op(ax, cx, cy, sym, r=0.22, fc='#ffffff', ec='#333333', fs=11):
    """Draw a circle operator (×, +, etc.)."""
    circ = plt.Circle((cx, cy), r, facecolor=fc, edgecolor=ec,
                      linewidth=1.4, zorder=5)
    ax.add_patch(circ)
    ax.text(cx, cy, sym, ha='center', va='center', fontsize=fs,
            fontweight='bold', color=ec, zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: General Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def draw_general_pipeline():
    # Canvas: wide enough for a skip line on the right side with clearance
    W, H = 7.0, 6.2          # inches at 300 dpi
    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    ax.set_xlim(0.8, 9.2)
    ax.set_ylim(2.9, 12.6)
    ax.axis('off')

    # ── colour palette ──
    C_IMG  = ('#e1f5fe', '#0288d1')   # light blue  – image / data boxes
    C_GA   = ('#e8f5e9', '#2e7d32')   # light green – GA boxes
    C_CNN  = ('#f3e5f5', '#7b1fa2')   # light purple – CNN / output
    C_OUT  = ('#ede7f6', '#4527a0')

    # Box parameters  (cx, cy, w, h, label, style)
    BW, BH = 5.0, 0.72   # default box width / height
    CX     = 5.5          # centre x for all main boxes

    boxes = [
        # (cy,  label,                         w,    h,    colour)
        (12.1, "Input Image\n256 × 256 × 3",            5.0, 0.65, C_IMG),
        (10.9, "Phase 1 – Feature Extraction\n8 Features  →  32 × 32 × 8 Tensor", 6.5, 0.65, C_IMG),
        ( 9.7, "Phase 2 – Genetic Algorithm (DEAP)\nEvolves Patch-Selection Rules", 6.5, 0.65, C_GA),
        ( 8.5, "Soft Attention Mask\n32 × 32 Grid  |  Sparsity ≈ 16 %",           6.0, 0.65, C_GA),
        ( 7.3, "Mask-Modulated Feature Tensor\n32 × 32 × 8",                       5.8, 0.65, C_IMG),
        ( 6.1, "Nearest-Neighbour Upsampling\n256 × 256 × 8 Feature Map",         5.8, 0.65, C_IMG),
        ( 4.7, "Phase 3 – CNN Classifier\n(Dual-Branch Attention Fusion)",         6.5, 0.75, C_CNN),
        ( 3.4, "AI-Generated vs. Human-Created\nClassification Output",            6.0, 0.65, C_OUT),
    ]

    # Draw boxes
    for (cy, txt, bw, bh, (fc, ec)) in boxes:
        rbox(ax, CX, cy, bw, bh, fc, ec)
        label(ax, CX, cy, txt, fs=9.5, color='#1a1a2e')

    # Straight arrows between consecutive boxes
    pairs = list(zip(boxes, boxes[1:]))
    # skip the gap between box[6] (CNN) and box[7] (output) – handled separately
    for i, ((cy1, *_, (fc1, ec1)), (cy2, *_, (fc2, ec2))) in enumerate(pairs):
        top_of_lower  = cy2 + boxes[i+1][3] / 2
        bot_of_upper  = cy1 - boxes[i][3] / 2
        # small gap for the skip line between Phase 3 and Output
        arrow(ax, CX, bot_of_upper, CX, top_of_lower)

    # ── Skip connection: raw image → CNN classifier ──
    # Drawn as a dashed line on the LEFT side, entirely within canvas
    SX   = 1.5                                  # x of the vertical part
    y_top  = 12.1                               # same row as "Input Image"
    y_bot  = 4.7 + 0.375                        # top edge of CNN box

    # Vertical dashed line
    ax.plot([SX, SX], [y_top, y_bot], color='#7b1fa2', lw=1.3, ls='--', zorder=5)
    # horizontal stub from Input Image left edge → skip line
    img_left = CX - 5.0 / 2
    ax.plot([img_left, SX], [y_top, y_top], color='#7b1fa2', lw=1.3, ls='--', zorder=5)
    # horizontal stub with arrow from skip line → CNN left edge
    cnn_left = CX - 6.5 / 2
    arrow(ax, SX, y_bot, cnn_left, y_bot, color='#7b1fa2', lw=1.3, style='->', ls='--')
    # label
    ax.text(SX - 0.3, (y_top + y_bot) / 2,
            "Raw Image\nSkip Connection", rotation=90,
            ha='center', va='center', fontsize=8.5,
            color='#7b1fa2', fontweight='bold', zorder=6)

    plt.savefig('paper/general_pipeline.png', bbox_inches='tight', dpi=300,
                facecolor='white')
    plt.close()
    print("Generated paper/general_pipeline.png successfully.")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: CNN Model Architecture
# ─────────────────────────────────────────────────────────────────────────────

def draw_model_architecture():
    W, H = 8.5, 7.5
    fig, ax = plt.subplots(figsize=(W, H), dpi=300)
    ax.set_xlim(0.0, 13.0)
    ax.set_ylim(3.6, 14.5)
    ax.axis('off')

    # ── palette ──
    C_IMG  = ('#e1f5fe', '#0288d1')
    C_FEAT = ('#fff3e0', '#e65100')
    C_GATE = ('#e8f5e9', '#2e7d32')
    C_JOIN = ('#f3e5f5', '#7b1fa2')
    C_OUT  = ('#ede7f6', '#4527a0')

    # Column x positions
    LX, RX = 3.5, 10.5    # left (image) branch, right (feature) branch
    MX = 7.0               # merge centre

    # ── Row 1: inputs ──
    rbox(ax, LX, 13.8, 4.0, 0.7, *C_IMG)
    label(ax, LX, 13.8, "Raw Image Input\n256 × 256 × 3", fs=9.5)

    rbox(ax, RX, 13.8, 4.0, 0.7, *C_FEAT)
    label(ax, RX, 13.8, "Upsampled Feature Map\n256 × 256 × 8", fs=9.5)

    # ── Arrows down from inputs ──
    arrow(ax, LX, 13.45, LX, 12.5)
    arrow(ax, RX, 13.45, RX, 12.5)

    # ── Row 2: branch CNNs ──
    rbox(ax, LX, 11.6, 4.2, 1.7, *C_IMG)
    label(ax, LX, 11.6,
          "Image Branch CNN\n3× Conv2D Blocks\n(32 → 64 → 128 Filters)\nBatchNorm + MaxPool", fs=9.0)

    rbox(ax, RX, 11.6, 4.2, 1.7, *C_FEAT)
    label(ax, RX, 11.6,
          "Feature Branch CNN\n3× Conv2D Blocks\n(32 → 64 → 128 Filters)\nBatchNorm + MaxPool", fs=9.0)

    img_bot  = 11.6 - 1.7 / 2   # = 10.75
    feat_bot = img_bot

    # ── Attention Gate: centred, below both branch boxes ──
    GATE_Y = 9.55
    rbox(ax, MX, GATE_Y, 4.2, 0.65, *C_GATE)
    label(ax, MX, GATE_Y, "Attention Gate\nConv2D 1×1 + Sigmoid", fs=8.8)

    # Curved arrow from Image branch bottom into Attention Gate top
    arrow(ax, LX, img_bot,  MX - 0.8, GATE_Y + 0.32,
          connectionstyle='arc3,rad=-0.15', color='#2e7d32')

    # ── Operator nodes: placed below the gate on their respective columns ──
    SP_Y = 8.3    # Spatial Gating × (right)
    CA_Y = 8.3    # Cross-Attention × (left)

    # Vertical arrows from branch bottoms to operators
    arrow(ax, LX, img_bot, LX, CA_Y + 0.28)
    arrow(ax, RX, feat_bot, RX, SP_Y + 0.28)

    # ── Spatial Gating ×  (right side) ──
    circle_op(ax, RX, SP_Y, '×', r=0.28, fc='#e8f5e9', ec='#2e7d32', fs=12)
    ax.text(RX + 0.55, SP_Y, "Spatial\nGating",
            ha='left', va='center', fontsize=8.5,
            color='#2e7d32', fontweight='bold')
    # Arrow: gate bottom-right corner → Spatial × (enters diagonally at top-left)
    arrow(ax, MX + 2.1, GATE_Y - 0.32, RX - 0.2, SP_Y + 0.2, color='#2e7d32')

    # ── Cross-Attention ×  (left side) ──
    circle_op(ax, LX, CA_Y, '×', r=0.28, fc='#f3e5f5', ec='#7b1fa2', fs=12)
    # Placed in the clear open space to the right of the left branch and above the horizontal arrow
    ax.text(LX + 0.55, CA_Y + 0.55, "Cross-Attention\nFusion",
            ha='left', va='center', fontsize=8.5,
            color='#7b1fa2', fontweight='bold')
    # Arrow: Spatial × → Cross-Attention × (horizontal)
    arrow(ax, RX - 0.28, SP_Y, LX + 0.28, CA_Y, color='#7b1fa2')

    # ── Residual + node (left column, below Cross-Attention ×) ──
    PLUS_Y = 7.05
    circle_op(ax, LX, PLUS_Y, '+', r=0.28, fc='#f3e5f5', ec='#7b1fa2', fs=12)
    # Arrow: Cross-Attention × → +
    arrow(ax, LX, CA_Y - 0.28, LX, PLUS_Y + 0.28)

    # Residual bypass: far-left vertical dashed line
    BYX = LX - 2.8
    ax.plot([LX - 2.1, BYX], [img_bot, img_bot],
            color='#7b1fa2', lw=1.3, ls='--', zorder=4)
    ax.plot([BYX, BYX], [img_bot, PLUS_Y],
            color='#7b1fa2', lw=1.3, ls='--', zorder=4)
    ax.annotate('', xy=(LX - 0.28, PLUS_Y), xytext=(BYX, PLUS_Y),
                arrowprops=dict(arrowstyle='->', color='#7b1fa2',
                                lw=1.3, linestyle='--'), zorder=5)
    ax.text(BYX - 0.25, (img_bot + PLUS_Y) / 2,
            "Residual\nBypass", ha='right', va='center',
            fontsize=8.0, color='#7b1fa2', fontweight='bold', rotation=90)

    # ── Joint Processing block ──
    JOINT_Y = 5.65
    rbox(ax, MX, JOINT_Y, 9.5, 0.85, *C_JOIN)
    label(ax, MX, JOINT_Y,
          "Joint Processing & Classification\n"
          "Conv2D (128) + GAP + Dropout (17 % / 34 %) + Dense (sigmoid)",
          fs=9.0)

    # Arrow: + straight down → Joint block
    arrow(ax, LX, PLUS_Y - 0.28, LX, JOINT_Y + 0.43, style='->')

    # ── Arrow: joint → output ──
    OUT_Y = 4.15
    arrow(ax, MX, JOINT_Y - 0.42, MX, OUT_Y + 0.31)

    # ── Output ──
    rbox(ax, MX, OUT_Y, 8.2, 0.62, *C_OUT)
    label(ax, MX, OUT_Y,
          "Classification Output – AI-Generated Probability  [0, 1]",
          fs=9.0, color='#4527a0', bold=True)

    plt.savefig('paper/model_architecture.png', bbox_inches='tight', dpi=300,
                facecolor='white')
    plt.close()
    print("Generated paper/model_architecture.png successfully.")


if __name__ == '__main__':
    draw_general_pipeline()
    draw_model_architecture()
