import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =============================================================================
# GEMMA 3 1B IT — SAE FEATURE ANALYSIS
# d_model = 1152 | d_latent = 2304 | k = 32 | Layer 13 activations
# =============================================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIVATIONS_DIR = "activations"
SAE_PATH = "temp/emotions.pt"
NUM_SAMPLES = 2  # must match what was used during collection

emotions = [
    "happy",
    "excited",
    "alert",
    "tense",
    "angry",
    "distressed",
    "sad",
    "depressed",
    "bored",
    "calm",
    "relaxed",
    "content",
]


class Gemma3TopKSAE(nn.Module):

    def __init__(self, d_model=1152, d_latent=2304, k=32):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_model, d_latent))
        self.b_enc = nn.Parameter(torch.zeros(d_latent))
        self.W_dec = nn.Parameter(torch.empty(d_latent, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def encode(self, x):
        latents_pre_act = (x - self.b_dec) @ self.W_enc + self.b_enc
        topk_values, topk_indices = torch.topk(latents_pre_act, self.k, dim=-1)
        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(-1, topk_indices, topk_values)
        return latents

    def decode(self, latents):
        return (latents @ self.W_dec) + self.b_dec

    def forward(self, x):
        latents = self.encode(x)
        return self.decode(latents), latents


def load_sae(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    cfg = checkpoint["cfg"]
    sae = Gemma3TopKSAE(d_model=cfg["d_model"], d_latent=cfg["d_latent"], k=cfg["k"]).to(DEVICE)
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.eval()
    dataset_mean = checkpoint["dataset_mean"].to(DEVICE)  # [1, d_model]
    neutral_unit = checkpoint["neutral_unit"].squeeze().to(DEVICE)  # [d_model], unit-norm
    print(f"  SAE config: d_model={cfg['d_model']}, d_latent={cfg['d_latent']}, k={cfg['k']}")
    print(f"  Dataset mean loaded (norm={dataset_mean.norm().item():.4f})")
    print(f"  Neutral unit loaded  (norm={neutral_unit.norm().item():.6f})")
    return sae, dataset_mean, neutral_unit


def load_all_activations(activations_dir):
    """Load activations per emotion, concatenating across all NUM_SAMPLES iterations."""
    pt_files = glob.glob(os.path.join(activations_dir, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {activations_dir}")

    buckets = {e: [] for e in emotions}
    for path in pt_files:
        stem = os.path.splitext(os.path.basename(path))[0]  # e.g. "happy_0"
        parts = stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        emotion, iteration = parts[0], int(parts[1])
        if emotion not in emotions or iteration >= NUM_SAMPLES:
            continue
        raw = torch.load(path, map_location=DEVICE)
        buckets[emotion].append(raw.view(-1, raw.shape[-1]))

    activations = {}
    for emotion, tensors in buckets.items():
        if not tensors:
            print(f"  WARNING: no files found for '{emotion}' — skipping")
            continue
        activations[emotion] = torch.cat(tensors, dim=0)
        print(f"  Loaded {emotion}: {activations[emotion].shape[0]} tokens")

    return activations


def verify_raw_orthogonality(activations):
    emotion_list = list(activations.keys())
    vectors = {e: activations[e].mean(dim=0) for e in emotion_list}

    col_w = max(len(e) for e in emotion_list) + 2
    print("\n--- Raw Space Cosine Similarity ---")
    print(f"{'':>{col_w}}" + "".join(f"{e:>{col_w}}" for e in emotion_list))
    for e_a in emotion_list:
        row = f"{e_a:>{col_w}}"
        for e_b in emotion_list:
            sim = F.cosine_similarity(vectors[e_a].unsqueeze(0),
                                      vectors[e_b].unsqueeze(0)).item()
            row += f"{sim:>{col_w}.4f}"
        print(row)


def verify_latent_orthogonality(sae, activations):
    emotion_list = list(activations.keys())
    latent_means = {e: sae.encode(activations[e]).mean(dim=0) for e in emotion_list}

    col_w = max(len(e) for e in emotion_list) + 2
    print("\n--- Latent Space Cosine Similarity ---")
    print(f"{'':>{col_w}}" + "".join(f"{e:>{col_w}}" for e in emotion_list))
    for e_a in emotion_list:
        row = f"{e_a:>{col_w}}"
        for e_b in emotion_list:
            sim = F.cosine_similarity(latent_means[e_a].unsqueeze(0),
                                      latent_means[e_b].unsqueeze(0)).item()
            row += f"{sim:>{col_w}.4f}"
        print(row)

    print("\n--- L0 Sparsity Per Emotion ---")
    for emotion, acts in activations.items():
        l0 = (sae.encode(acts) > 0).float().sum(dim=-1).mean().item()
        print(f"  {emotion}: {l0:.1f} active features (target k={sae.k})")


def check_feature_disjointness(sae, activations):
    emotion_list = list(activations.keys())
    fire_masks = {e: (sae.encode(activations[e]) > 0).any(dim=0) for e in emotion_list}

    print("\n--- Pairwise Feature Disjointness ---")
    for i, e_a in enumerate(emotion_list):
        for e_b in emotion_list[i + 1:]:
            shared = (fire_masks[e_a] & fire_masks[e_b]).sum().item()
            unique_a = fire_masks[e_a].sum().item() - shared
            unique_b = fire_masks[e_b].sum().item() - shared
            print(f"  {e_a} vs {e_b}: "
                  f"shared={shared:.0f} | "
                  f"unique_{e_a}={unique_a:.0f} | "
                  f"unique_{e_b}={unique_b:.0f}")


def verify_unique_feature_similarity(sae, activations):
    """Cosine similarity computed only on features exclusive to each pair (XOR mask)."""
    emotion_list = list(activations.keys())
    fire_masks = {e: (sae.encode(activations[e]) > 0).any(dim=0) for e in emotion_list}
    latent_means = {e: sae.encode(activations[e]).mean(dim=0) for e in emotion_list}

    col_w = max(len(e) for e in emotion_list) + 2
    print("\n--- Unique-Feature Cosine Similarity (exclusive dims only) ---")
    print(f"{'':>{col_w}}" + "".join(f"{e:>{col_w}}" for e in emotion_list))
    for e_a in emotion_list:
        row = f"{e_a:>{col_w}}"
        for e_b in emotion_list:
            if e_a == e_b:
                row += f"{'1.0000':>{col_w}}"
                continue
            exclusive = fire_masks[e_a] ^ fire_masks[e_b]  # fires in one but not both
            if exclusive.sum() == 0:
                row += f"{'  N/A':>{col_w}}"
                continue
            v_a = latent_means[e_a][exclusive].unsqueeze(0)
            v_b = latent_means[e_b][exclusive].unsqueeze(0)
            sim = F.cosine_similarity(v_a, v_b).item()
            row += f"{sim:>{col_w}.4f}"
        print(row)


def plot_latent_projections(sae, activations, save_path="images/latent_projections.png"):
    """PCA + t-SNE side-by-side of per-emotion mean latent vectors (12 points each)."""
    emotion_list = list(activations.keys())
    colors = plt.cm.tab20.colors[:len(emotion_list)]

    mean_latents = np.stack([
        sae.encode(activations[e]).mean(dim=0).cpu().numpy()
        for e in emotion_list
    ])  # [12, d_latent]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: PCA
    pca = PCA(n_components=2)
    coords_pca = pca.fit_transform(mean_latents)
    ax = axes[0]
    for i, emotion in enumerate(emotion_list):
        ax.scatter(coords_pca[i, 0], coords_pca[i, 1], color=colors[i], s=120, zorder=5)
        ax.annotate(emotion, (coords_pca[i, 0], coords_pca[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    var = pca.explained_variance_ratio_.sum() * 100
    ax.set_title(f"PCA  ({var:.1f}% variance explained)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.grid(True, linestyle="--", alpha=0.4)

    # Right: t-SNE (perplexity must be < n_samples; 5 is safe for 12 points)
    tsne = TSNE(n_components=2, perplexity=5, random_state=42, max_iter=1000)
    coords_tsne = tsne.fit_transform(mean_latents)
    ax = axes[1]
    for i, emotion in enumerate(emotion_list):
        ax.scatter(coords_tsne[i, 0], coords_tsne[i, 1], color=colors[i], s=120, zorder=5)
        ax.annotate(emotion, (coords_tsne[i, 0], coords_tsne[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.set_title("t-SNE  (perplexity=5)")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True, linestyle="--", alpha=0.4)

    fig.suptitle("SAE Latent Emotion Projections — Gemma 3 1B IT (Layer 13)", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print()
    print(f"Projection plot saved to: {save_path}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _build_fire_masks(sae, activations):
    return {e: (sae.encode(activations[e]) > 0).any(dim=0) for e in activations}


def _raw_sim_matrix(activations):
    emotion_list = list(activations.keys())
    vectors = {e: activations[e].mean(dim=0) for e in emotion_list}
    n = len(emotion_list)
    mat = np.zeros((n, n))
    for i, e_a in enumerate(emotion_list):
        for j, e_b in enumerate(emotion_list):
            mat[i, j] = F.cosine_similarity(
                vectors[e_a].unsqueeze(0), vectors[e_b].unsqueeze(0)).item()
    return mat, emotion_list


def _latent_sim_matrix(sae, activations):
    emotion_list = list(activations.keys())
    means = {e: sae.encode(activations[e]).mean(dim=0) for e in emotion_list}
    n = len(emotion_list)
    mat = np.zeros((n, n))
    for i, e_a in enumerate(emotion_list):
        for j, e_b in enumerate(emotion_list):
            mat[i, j] = F.cosine_similarity(
                means[e_a].unsqueeze(0), means[e_b].unsqueeze(0)).item()
    return mat, emotion_list


def _unique_sim_matrix(sae, activations):
    emotion_list = list(activations.keys())
    fire_masks = _build_fire_masks(sae, activations)
    means = {e: sae.encode(activations[e]).mean(dim=0) for e in emotion_list}
    n = len(emotion_list)
    mat = np.full((n, n), np.nan)
    for i, e_a in enumerate(emotion_list):
        mat[i, i] = 1.0
        for j, e_b in enumerate(emotion_list):
            if i == j:
                continue
            exclusive = fire_masks[e_a] ^ fire_masks[e_b]
            if exclusive.sum() == 0:
                continue
            mat[i, j] = F.cosine_similarity(
                means[e_a][exclusive].unsqueeze(0),
                means[e_b][exclusive].unsqueeze(0)).item()
    return mat, emotion_list


# ─── Cosine similarity heatmaps ───────────────────────────────────────────────

def plot_similarity_heatmap(matrix, labels, title, save_path, symmetric=True):
    n = len(labels)
    cell = 1.4          # inches per cell — keeps cells square
    fig, ax = plt.subplots(figsize=(n * cell + 2.5, n * cell + 1.5))

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad(color="#cccccc")

    if symmetric:
        vmax = np.nanmax(np.abs(matrix))
        vmin = -vmax
    else:
        valid_vals = matrix[~np.isnan(matrix)]
        vmin, vmax = float(valid_vals.min()), float(valid_vals.max())
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.set_xticklabels(labels, rotation=45, ha="left", fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)

    for i in range(n):
        for j in range(n):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=9, color="#777777")
            else:
                norm_val = (val - vmin) / (vmax - vmin + 1e-9)
                r, g, b, _ = cmap(norm_val)
                luminance = 0.299 * r + 0.587 * g + 0.114 * b
                text_color = "black" if luminance > 0.45 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=9, color=text_color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Heatmap saved to: {save_path}")


def plot_all_similarity_heatmaps(sae, activations):
    mat, labels = _raw_sim_matrix(activations)
    plot_similarity_heatmap(mat, labels,
        "Raw Space Cosine Similarity — Centered Activations",
        "images/heatmap_raw_similarity.png")
    mat, labels = _latent_sim_matrix(sae, activations)
    plot_similarity_heatmap(mat, labels,
        "Latent Space Cosine Similarity — SAE Encoded",
        "images/heatmap_latent_similarity.png", symmetric=False)
    mat, labels = _unique_sim_matrix(sae, activations)
    plot_similarity_heatmap(mat, labels,
        "Unique-Feature Cosine Similarity — Exclusive Dims Only",
        "images/heatmap_unique_similarity.png")


# ─── Fingerprint heatmap ──────────────────────────────────────────────────────

def plot_fingerprint_heatmap(sae, activations, top_n=10, save_path="images/fingerprint_heatmap.png"):
    # Circumplex order: use the global emotions list as the canonical column sequence
    emotion_list = [e for e in emotions if e in activations]
    n_emo = len(emotion_list)
    emo_idx = {e: i for i, e in enumerate(emotion_list)}

    latents = {e: sae.encode(activations[e]) for e in emotion_list}
    peak = {e: latents[e].max(dim=0).values for e in emotion_list}

    # Top-N dominant features per emotion: peak ≥ 1.5× peak of any other emotion
    rows = []          # (emotion, feat_idx, peak_mag)
    group_sizes = {}

    for emotion in emotion_list:
        peak_others = torch.stack([peak[e] for e in emotion_list if e != emotion]).max(dim=0).values
        unique_mask = peak[emotion] >= 1.5 * peak_others
        if unique_mask.sum() == 0:
            group_sizes[emotion] = 0
            continue

        peak_mags = peak[emotion][unique_mask]
        unique_indices = torch.where(unique_mask)[0]
        sorted_mags, sorted_idx = torch.sort(peak_mags, descending=True)

        count = min(top_n, len(sorted_idx))
        group_sizes[emotion] = count
        for i in range(count):
            feat_idx = unique_indices[sorted_idx[i]].item()
            peak_mag = sorted_mags[i].item()
            rows.append((emotion, feat_idx, peak_mag))

    if not rows:
        print("No exclusive features found — skipping fingerprint heatmap.")
        return

    n_rows = len(rows)

    # Build data matrix [n_rows, n_emo] — non-zero only in the owning emotion column
    data = np.zeros((n_rows, n_emo))
    for r_idx, (emotion, feat_idx, peak_mag) in enumerate(rows):
        data[r_idx, emo_idx[emotion]] = peak_mag

    # Adaptive sizing — row height scales so text stays legible
    col_w = 1.5
    row_h = max(0.32, min(0.7, 18.0 / max(n_rows, 1)))
    cell_fs = max(5, min(8, int(row_h * 15)))
    fig_w = n_emo * col_w + 3.0
    fig_h = n_rows * row_h + 2.5

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    cmap = plt.cm.Blues
    im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0)

    # x-axis: emotion names at top in circumplex order
    ax.set_xticks(range(n_emo))
    ax.set_xticklabels(emotion_list, rotation=45, ha="left", fontsize=10)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    # y-axis: SAE feature index per row
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([str(r[1]) for r in rows], fontsize=8)
    ax.set_ylabel(f"SAE feature index (top-{top_n} exclusive per emotion, by peak magnitude)",
                  fontsize=9)

    # Cell annotations — 3 decimal places, luminance-aware text colour
    vmax = float(data.max()) + 1e-9
    for i in range(n_rows):
        for j in range(n_emo):
            val = data[i, j]
            r, g, b, _ = cmap(val / vmax)
            lum = 0.299 * r + 0.587 * g + 0.114 * b
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=cell_fs, color="black" if lum > 0.45 else "white")

    # Horizontal dividers between emotion groups
    cumulative = 0
    for emotion in emotion_list:
        cumulative += group_sizes.get(emotion, 0)
        if 0 < cumulative < n_rows:
            ax.axhline(cumulative - 0.5, color="#888888", linewidth=1.0, alpha=0.6)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03, label="Peak activation magnitude")
    ax.set_title(
        f"Emotion Feature Fingerprint — Top {top_n} Exclusive Features by Peak Magnitude (Layer 13)",
        fontsize=12, pad=40)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Fingerprint heatmap saved to: {save_path}")


# ─── Shared / Exclusive feature profiles ─────────────────────────────────────

def plot_feature_profile(sae, activations, mode="exclusive", save_path=None):
    """
    Left panel : horizontal bar showing feature count per emotion.
    Right panel: Gaussian (μ, σ) of those features' activation values,
                 one curve per emotion stacked vertically.
    mode = 'exclusive' | 'shared'
    """
    emotion_list = list(activations.keys())
    n = len(emotion_list)
    colors = list(plt.cm.tab20.colors[:n])

    latents = {e: sae.encode(activations[e]) for e in emotion_list}
    fire_masks = {e: (latents[e] > 0).any(dim=0) for e in emotion_list}
    peak = {e: latents[e].max(dim=0).values for e in emotion_list}

    profiles = []
    for emotion in emotion_list:
        if mode == "exclusive":
            peak_others = torch.stack([peak[e] for e in emotion_list if e != emotion]).max(dim=0).values
            mask = peak[emotion] >= 1.5 * peak_others
        else:
            others = torch.zeros(sae.d_latent, dtype=torch.bool, device=DEVICE)
            for other in emotion_list:
                if other != emotion:
                    others |= fire_masks[other]
            mask = fire_masks[emotion] & others

        count = int(mask.sum().item())
        mu, sigma = None, None
        if count > 0:
            vals = latents[emotion][:, mask].cpu().numpy().flatten()
            vals = vals[vals > 0]
            if len(vals) >= 2:
                mu, sigma = float(vals.mean()), float(vals.std())
            elif len(vals) == 1:
                mu, sigma = float(vals[0]), 0.0
        profiles.append({"emotion": emotion, "count": count, "mu": mu, "sigma": sigma})

    fig = plt.figure(figsize=(14, max(7, n * 0.8)))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 2.5], wspace=0.0)
    ax_bar = fig.add_subplot(gs[0])
    ax_gauss = fig.add_subplot(gs[1], sharey=ax_bar)

    y_pos = np.arange(n)
    row_height = 0.75

    # Bar panel
    counts = [p["count"] for p in profiles]
    ax_bar.barh(y_pos, counts, color=colors, edgecolor="white", height=row_height)
    max_count = max(counts) or 1
    for i, c in enumerate(counts):
        ax_bar.text(c + max_count * 0.03, y_pos[i], str(c), va="center", fontsize=8)
    ax_bar.set_xlim(right=max_count * 1.25)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(emotion_list, fontsize=10)
    ax_bar.set_xlabel("Feature count", fontsize=9)
    ax_bar.grid(axis="x", linestyle="--", alpha=0.3)
    ax_bar.spines["right"].set_visible(False)
    ax_bar.spines["top"].set_visible(False)

    # Gaussian panel
    valid = [(p["mu"], p["sigma"]) for p in profiles
             if p["mu"] is not None and p["sigma"] is not None and p["sigma"] > 0]

    if valid:
        x_min = min(mu - 3.5 * sig for mu, sig in valid)
        x_max = max(mu + 3.5 * sig for mu, sig in valid)
        x = np.linspace(x_min, x_max, 400)

        for i, p in enumerate(profiles):
            y_base = y_pos[i] - row_height / 2
            if p["mu"] is None or p["sigma"] is None:
                ax_gauss.text(
                    (x_min + x_max) / 2, y_pos[i], "no features",
                    ha="center", va="center", fontsize=8,
                    color="#999999", style="italic")
                continue
            if p["sigma"] <= 0:
                ax_gauss.plot(
                    [p["mu"], p["mu"]],
                    [y_pos[i] - row_height * 0.4, y_pos[i] + row_height * 0.4],
                    color=colors[i], linewidth=2)
                continue

            curve = _gaussian_pdf(x, p["mu"], p["sigma"])
            curve_scaled = y_base + curve / curve.max() * row_height
            ax_gauss.plot(x, curve_scaled, color=colors[i], linewidth=1.8)
            ax_gauss.fill_between(x, y_base, curve_scaled,
                                  color=colors[i], alpha=0.25)
            # μ marker
            ax_gauss.plot([p["mu"], p["mu"]],
                          [y_base, y_base + row_height * 0.9],
                          color=colors[i], linewidth=0.9,
                          linestyle="--", alpha=0.7)

        # μ & σ annotations at right edge (axes x-coord, data y-coord)
        trans = ax_gauss.get_yaxis_transform()
        for i, p in enumerate(profiles):
            if p["mu"] is not None:
                lbl = (f"μ={p['mu']:.2f}, σ={p['sigma']:.2f}"
                       if p["sigma"] is not None and p["sigma"] > 0
                       else f"μ={p['mu']:.2f}")
                ax_gauss.text(0.98, y_pos[i], lbl, transform=trans,
                              va="center", ha="right", fontsize=7.5, color=colors[i])

    ax_gauss.set_xlabel("Activation magnitude", fontsize=9)
    ax_gauss.set_ylim(-0.5, n - 0.5)
    ax_gauss.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax_gauss.grid(axis="x", linestyle="--", alpha=0.3)
    ax_gauss.spines["left"].set_visible(False)
    ax_gauss.spines["top"].set_visible(False)

    mode_label = "Exclusive" if mode == "exclusive" else "Shared"
    fig.suptitle(
        f"{mode_label} Feature Profile — Gemma 3 1B IT (Layer 13)", fontsize=12)
    if save_path is None:
        save_path = f"images/{mode}_feature_profile.png"
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{mode_label} feature profile saved to: {save_path}")


# ─── Reconstruction error ─────────────────────────────────────────────────────

def plot_reconstruction_error(sae, activations, save_path="images/reconstruction_error.png"):
    emotion_list = list(activations.keys())
    errors = {}
    for emotion in emotion_list:
        acts = activations[emotion]
        recon, _ = sae(acts)
        errors[emotion] = F.mse_loss(recon, acts).item()

    global_mse = float(np.mean(list(errors.values())))

    # Sort ascending by deviation so best (most negative) is at bottom, worst at top
    sorted_emos = sorted(emotion_list, key=lambda e: errors[e] - global_mse)
    deviations = [errors[e] - global_mse for e in sorted_emos]
    n = len(sorted_emos)

    x_abs = max(abs(d) for d in deviations)
    head_len = x_abs * 0.08
    head_w = 0.35

    fig, ax = plt.subplots(figsize=(9, max(6, n * 0.6)))
    ax.set_xlim(-(x_abs * 1.45), x_abs * 1.45)
    ax.set_ylim(-1, n)
    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_emos, fontsize=10)

    for i, (emotion, dev) in enumerate(zip(sorted_emos, deviations)):
        color = "#347768" if dev < 0 else ("#6B273D" if dev > 0 else "black")
        if abs(dev) > 1e-6:
            ax.arrow(0, i, dev, 0,
                     head_width=head_w,
                     head_length=head_len,
                     width=0.18,
                     fc=color,
                     ec=color,
                     length_includes_head=True)
        # Actual MSE label at arrowhead
        offset = head_len * 1.2 * (1 if dev >= 0 else -1)
        ax.text(dev + offset, i, f"{errors[emotion]:.3f}",
                va="center", ha="left" if dev >= 0 else "right",
                fontsize=8.5, color=color)

    ax.axvline(x=0, color="0.75", ls="--", lw=1.5, zorder=0,
               label=f"Mean MSE = {global_mse:.3f}")
    ax.grid(axis="y", color="0.92")
    ax.set_xlabel("Deviation from mean MSE", fontsize=10)
    ax.set_title("Reconstruction Error per Emotion — Gemma 3 1B IT (Layer 13)", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Reconstruction error plot saved to: {save_path}")


def extract_high_magnitude_uniques(sae, activations, top_n=10):
    emotion_list = list(activations.keys())
    latents = {e: sae.encode(activations[e]) for e in emotion_list}

    peak = {e: latents[e].max(dim=0).values for e in emotion_list}

    print(f"\n--- Top {top_n} Dominant Features Per Emotion (peak ≥ 1.5× any other emotion) ---")
    for emotion in emotion_list:
        peak_others = torch.stack([peak[e] for e in emotion_list if e != emotion]).max(dim=0).values
        unique_mask = peak[emotion] >= 1.5 * peak_others

        if unique_mask.sum() == 0:
            print(f"\n  {emotion}: no dominant features found")
            continue

        peak_mags = peak[emotion][unique_mask]
        unique_indices = torch.where(unique_mask)[0]
        sorted_mags, sorted_idx = torch.sort(peak_mags, descending=True)

        print(f"\n  {emotion}:")
        for i in range(min(top_n, len(sorted_idx))):
            idx = unique_indices[sorted_idx[i]].item()
            mag = sorted_mags[i].item()
            print(f"    Feature {idx:5d} | Peak Magnitude: {mag:.4f}")


if __name__ == "__main__":
    print("Loading SAE...")
    sae, dataset_mean, neutral_unit = load_sae(SAE_PATH)

    print("\nLoading activations...")
    activations = load_all_activations(ACTIVATIONS_DIR)

    # Apply the same centering + orthogonalization used during SAE training
    activations = {e: acts - dataset_mean for e, acts in activations.items()}
    print("  Centering applied to all emotion activations.")
    activations = {
        e: acts - (acts @ neutral_unit).unsqueeze(-1) * neutral_unit
        for e, acts in activations.items()
    }
    print("  Neutral orthogonalization applied — activations are now pure emotion vectors.")

    with torch.no_grad():
        
        verify_raw_orthogonality(activations)
        verify_latent_orthogonality(sae, activations)
        verify_unique_feature_similarity(sae, activations)
        check_feature_disjointness(sae, activations)
        extract_high_magnitude_uniques(sae, activations)
        plot_latent_projections(sae, activations)
        plot_all_similarity_heatmaps(sae, activations)
        plot_fingerprint_heatmap(sae, activations)
        plot_feature_profile(sae, activations, mode="exclusive",
                             save_path="images/exclusive_feature_profile.png")
        plot_feature_profile(sae, activations, mode="shared",
                             save_path="images/shared_feature_profile.png")
        plot_reconstruction_error(sae, activations)

    # Save activations as vector.pt
    os.makedirs("temp", exist_ok=True)
    os.makedirs("images", exist_ok=True)
    torch.save(activations, "temp/emotion_vectors.pt")
    print("Activations saved to temp/emotion_vectors.pt")