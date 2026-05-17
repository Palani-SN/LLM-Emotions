"""
Extract the two dominant axes of emotion space from activation vectors.

Method: PCA (via SVD) on the 12 emotion mean vectors.
        No hardcoded labels or circumplex coordinates — axes emerge purely
        from the geometry of the activations.  The top two PCs capture the
        dominant directions of inter-emotion variation and empirically
        correspond to valence (PC1) and arousal (PC2).

Output: steering_vectors.pt
  {
    "pc1":                      (D,)   dominant axis,
    "pc2":                      (D,)   second axis, orthogonal to pc1,
    "valence":                  (D,)   unit-norm valence direction (sign-oriented: happy > sad),
    "arousal":                  (D,)   unit-norm arousal direction (sign-oriented: excited > calm),
    "projections":              (N, 2) each emotion's (pc1, pc2) coordinates,
    "mean_vecs":                (N, D) centered mean activation per emotion,
    "emotions":                 list[str],
    "explained_variance_ratio": (2,)   fraction of inter-emotion variance per axis,
  }
"""

import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def assign_valence_arousal(pc1: torch.Tensor, pc2: torch.Tensor,
                           projections: torch.Tensor, emotions: list):
    """
    Identify which of pc1/pc2 is the valence axis and which is arousal,
    and orient their signs — using only the emotion names already in the
    dataset as reference pairs, no external coordinates.

    Valence reference : 'happy' should project positive, 'sad' negative.
    Arousal reference : 'excited' should project positive, 'calm' negative.

    Returns (valence_vec, arousal_vec) both unit-norm.
    """
    idx = {e: i for i, e in enumerate(emotions)}

    sep_on_pc1 = {
        "valence": (projections[idx["happy"], 0] - projections[idx["sad"], 0]).item(),
        "arousal": (projections[idx["excited"], 0] - projections[idx["calm"], 0]).item(),
    }
    sep_on_pc2 = {
        "valence": (projections[idx["happy"], 1] - projections[idx["sad"], 1]).item(),
        "arousal": (projections[idx["excited"], 1] - projections[idx["calm"], 1]).item(),
    }

    # Whichever PC gives a larger absolute separation for happy/sad → valence axis
    if abs(sep_on_pc1["valence"]) >= abs(sep_on_pc2["valence"]):
        valence_vec = pc1 * (1 if sep_on_pc1["valence"] > 0 else -1)
        arousal_vec = pc2 * (1 if sep_on_pc2["arousal"] > 0 else -1)
        val_sep = sep_on_pc1["valence"]
        aro_sep = sep_on_pc2["arousal"]
        print("\nAxis assignment:  PC1 → valence  |  PC2 → arousal")
    else:
        valence_vec = pc2 * (1 if sep_on_pc2["valence"] > 0 else -1)
        arousal_vec = pc1 * (1 if sep_on_pc1["arousal"] > 0 else -1)
        val_sep = sep_on_pc2["valence"]
        aro_sep = sep_on_pc1["arousal"]
        print("\nAxis assignment:  PC2 → valence  |  PC1 → arousal  (axes swapped)")

    print(f"  happy−sad    separation on valence axis : {val_sep:.4f}")
    print(f"  excited−calm separation on arousal axis : {aro_sep:.4f}")

    return valence_vec, arousal_vec


def plot_emotion_vectors(projections: np.ndarray, emotions: list, evr: np.ndarray,
                         valence_2d: np.ndarray, arousal_2d: np.ndarray,
                         save_path: str = "images/emotion_vectors.png") -> None:
    colors = list(plt.cm.tab20.colors[:len(emotions)])

    # Variance-normalise each axis independently so neither PC dominates visually.
    # This only affects the plot — steering_vectors.pt keeps raw projections.
    std1 = projections[:, 0].std() or 1.0
    std2 = projections[:, 1].std() or 1.0
    proj = projections / np.array([std1, std2])

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.axhline(0, color="black", linewidth=0.8, zorder=0)
    ax.axvline(0, color="black", linewidth=0.8, zorder=0)

    pad = max(np.abs(proj).max() * 1.35, 0.01)
    ax.set_xlim(-pad, pad)
    ax.set_ylim(-pad, pad)

    # ── Emotion arrows ────────────────────────────────────────────────────────
    for i, (emotion, color) in enumerate(zip(emotions, colors)):
        x, y = proj[i]
        ax.annotate(
            "",
            xy=(x, y), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=color, lw=2.0),
        )

    # ── Valence & arousal axes — scaled to 85 % of plot range ────────────────
    display_len = pad * 0.85
    for vec_2d, label in [(valence_2d, "valence"), (arousal_2d, "arousal")]:
        # Apply same variance normalisation, then rescale to fixed display length
        v_norm = vec_2d / np.array([std1, std2])
        mag = np.linalg.norm(v_norm)
        if mag > 0:
            v_scaled = v_norm / mag * display_len
        else:
            v_scaled = v_norm
        ax.annotate(
            "",
            xy=(v_scaled[0], v_scaled[1]), xytext=(0, 0),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=2.2,
                            linestyle="dashed",
                            connectionstyle="arc3,rad=0.0"),
            zorder=5,
        )
        # Label offset slightly beyond the arrowhead
        offset = v_scaled * 1.08
        ax.text(offset[0], offset[1], label,
                fontsize=10, ha="center", va="center",
                fontweight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black",
                          lw=0.8, alpha=0.85))

    # ── Corner legends: group emotions by quadrant ────────────────────────────
    quadrants: dict = {(1, 1): [], (1, -1): [], (-1, 1): [], (-1, -1): []}
    for i, emotion in enumerate(emotions):
        qx = 1 if proj[i, 0] >= 0 else -1
        qy = 1 if proj[i, 1] >= 0 else -1
        quadrants[(qx, qy)].append((emotion, colors[i]))

    # (corner_x, base_y, ha, dy_per_line)  — all in axes-fraction coordinates
    corner_cfg = {
        ( 1,  1): (0.98, 0.97, "right", -0.055),   # top-right,    stack down
        (-1,  1): (0.02, 0.97, "left",  -0.055),   # top-left,     stack down
        (-1, -1): (0.02, 0.03, "left",  +0.055),   # bottom-left,  stack up
        ( 1, -1): (0.98, 0.03, "right", +0.055),   # bottom-right, stack up
    }

    for (qx, qy), members in quadrants.items():
        cx, base_y, ha, dy = corner_cfg[(qx, qy)]
        for j, (emotion, color) in enumerate(members):
            ax.text(
                cx, base_y + j * dy, f"● {emotion}",
                transform=ax.transAxes,
                fontsize=8, ha=ha, va="center",
                color=color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )

    ax.set_xlabel(f"PC1  ({evr[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2  ({evr[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("Emotion vectors in PC1–PC2 space  (variance-normalised axes)", fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot → {save_path}")


def main():
    os.makedirs("temp", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    print("Loading activations from vectors.pt ...")
    activations: dict = torch.load("temp/emotion_vectors.pt", map_location=DEVICE)

    emotions = list(activations.keys())
    print(f"  Emotions ({len(emotions)}): {emotions}")

    # ── Mean vector per emotion: (N, D) ───────────────────────────────────────
    mean_vecs = torch.stack([activations[e].mean(dim=0) for e in emotions])

    # ── Centre across emotions — PCA captures inter-emotion variance only ─────
    grand_mean = mean_vecs.mean(dim=0, keepdim=True)
    X = mean_vecs - grand_mean  # (N, D)

    # ── PCA via full SVD: X = U S Vh  →  rows of Vh are principal directions ──
    _, S, Vh = torch.linalg.svd(X, full_matrices=False)  # Vh: (N, D)

    total_var = (S ** 2).sum()
    evr = (S ** 2) / total_var  # explained variance ratio per component

    print("\nExplained variance by component:")
    for i in range(min(4, len(S))):
        print(f"  PC{i+1}: {evr[i].item()*100:.1f}%")

    pc1 = Vh[0]  # (D,) — dominant axis
    pc2 = Vh[1]  # (D,) — already orthogonal to pc1 by SVD construction

    # ── Verify orthogonality ───────────────────────────────────────────────────
    dot = (pc1 @ pc2).item()
    print(f"\nPC1 · PC2 = {dot:.2e}  (should be ~0)")
    print(f"||PC1|| = {pc1.norm().item():.6f}  ||PC2|| = {pc2.norm().item():.6f}")

    # ── Project each emotion onto the two axes ─────────────────────────────────
    projections = X @ torch.stack([pc1, pc2], dim=1)  # (N, 2)

    print("\nEmotion projections in (PC1, PC2) space:")
    print(f"  {'emotion':<14}  {'PC1':>9}  {'PC2':>9}")
    print(f"  {'-'*14}  {'-'*9}  {'-'*9}")
    for i, emotion in enumerate(emotions):
        p1 = projections[i, 0].item()
        p2 = projections[i, 1].item()
        print(f"  {emotion:<14}  {p1:>9.4f}  {p2:>9.4f}")

    # ── Identify valence / arousal from the PCs ───────────────────────────────
    valence_vec, arousal_vec = assign_valence_arousal(pc1, pc2, projections, emotions)
    print(f"  ||valence|| = {valence_vec.norm().item():.6f}")
    print(f"  ||arousal|| = {arousal_vec.norm().item():.6f}")
    print(f"  valence · arousal = {(valence_vec @ arousal_vec).item():.2e}")

    # ── 4-quadrant emotion vector plot ────────────────────────────────────────
    # Project valence/arousal unit vectors into the (pc1, pc2) plane for plotting
    val_2d = np.array([(valence_vec @ pc1).item(), (valence_vec @ pc2).item()])
    aro_2d = np.array([(arousal_vec @ pc1).item(), (arousal_vec @ pc2).item()])
    plot_emotion_vectors(projections.cpu().numpy(), emotions, evr.cpu().numpy(),
                         val_2d, aro_2d)

    # ── Save ───────────────────────────────────────────────────────────────────
    torch.save(
        {
            "pc1":                      pc1,
            "pc2":                      pc2,
            "valence":                  valence_vec,
            "arousal":                  arousal_vec,
            "projections":              projections,
            "mean_vecs":                mean_vecs,
            "emotions":                 emotions,
            "explained_variance_ratio": evr[:2],
        },
        "temp/steering_vectors.pt",
    )
    print("\nSaved temp/steering_vectors.pt")


if __name__ == "__main__":
    main()
