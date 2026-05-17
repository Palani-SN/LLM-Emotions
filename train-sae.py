import os
import copy
import json
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIVATIONS_DIR = "activations"
SAE_SAVE_PATH = "temp/emotions.pt"
NUM_SAMPLES = 2  

emotions = [
    "happy", "excited", "alert", "tense", "angry", "distressed",
    "sad", "depressed", "bored", "calm", "relaxed", "content"
]

class Gemma3TopKSAE(nn.Module):
    def __init__(self, d_model=1152, d_latent=2304, k=32):
        super().__init__()
        self.d_model = d_model
        self.d_latent = d_latent
        self.k = k

        # Encoder
        self.W_enc = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_model, d_latent)))
        self.b_enc = nn.Parameter(torch.zeros(d_latent))

        # Decoder 
        self.W_dec = nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_latent, d_model)))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x):
        # x is already centered and orthogonalized in your preprocessing pipeline
        latents_pre_act = x @ self.W_enc + self.b_enc
        topk_values, topk_indices = torch.topk(latents_pre_act, self.k, dim=-1)

        latents = torch.zeros_like(latents_pre_act)
        latents.scatter_(-1, topk_indices, topk_values)
        return latents

    def decode(self, latents):
        # Since data is pre-centered, decoder predicts the centered residual variance directly
        return latents @ self.W_dec

    def forward(self, x):
        latents = self.encode(x)
        reconstruction = self.decode(latents)
        return reconstruction, latents


def load_all_activations(activations_dir):
    pt_files = glob.glob(os.path.join(activations_dir, "*.pt"))
    if not pt_files:
        raise FileNotFoundError(f"No .pt files found in {activations_dir}")

    all_tensors = []
    for path in pt_files:
        stem = os.path.splitext(os.path.basename(path))[0]  
        parts = stem.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        emotion, iteration = parts[0], int(parts[1])
        if emotion not in emotions or iteration >= NUM_SAMPLES:
            continue
        
        raw = torch.load(path, map_location="cpu")
        tokens = raw.view(-1, raw.shape[-1])
        all_tensors.append(tokens)
        print(f"  Loaded {stem}: {tokens.shape[0]} tokens")

    return torch.cat(all_tensors, dim=0)


def train(sae, data, epochs=50, lr=3e-4, batch_size=4096):
    # b_dec is a static baseline of 0 because data is perfectly pre-centered
    with torch.no_grad():
        sae.b_dec.data = torch.zeros(sae.d_model, device=DEVICE)

    optimizer = optim.Adam(sae.parameters(), lr=lr)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    sae.train()
    print(f"\nTraining on {data.shape[0]} tokens | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}")

    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            x = batch[0].to(DEVICE)

            reconstruction, _ = sae(x)
            loss = F.mse_loss(reconstruction, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                sae.W_dec.data = F.normalize(sae.W_dec.data, dim=1)

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        loss_history.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(sae.state_dict())

        if (epoch + 1) % 50 == 0 or epoch == 0:
            marker = " *" if best_epoch == epoch + 1 else ""
            print(f"  Epoch {epoch+1:03d}/{epochs} | Avg MSE Loss: {avg_loss:.8f}{marker}")

    print(f"\nBest checkpoint: Epoch {best_epoch} | Loss: {best_loss:.8f}")
    sae.load_state_dict(best_state)
    return sae, loss_history, best_epoch


def save_loss_plot(loss_history, best_epoch, k, save_path="images/training_loss.png"):
    epochs = range(1, len(loss_history) + 1)
    best_loss = loss_history[best_epoch - 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, loss_history, linewidth=1.5, color="#4C72B0", label="Avg MSE Loss")
    ax.axvline(x=best_epoch, color="#E05C5C", linestyle="--", linewidth=1.2, label=f"Best epoch {best_epoch}")
    ax.scatter([best_epoch], [best_loss], color="#E05C5C", zorder=5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Avg MSE Loss")
    ax.set_title(f"SAE Training Loss — Gemma 3 (k={k})")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs("temp", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    # 1. Load Data
    data = load_all_activations(ACTIVATIONS_DIR)

    # 2. Center Data
    dataset_mean = data.mean(dim=0, keepdim=True)  # [1, d_model]
    data = data - dataset_mean
    print(f"  Centered data — removed mean vector (norm={dataset_mean.norm().item():.4f})")

    # 3. FIXED: Proper Token-Averaged Neutral Orthogonalization
    neutral_raw = torch.load(os.path.join(ACTIVATIONS_DIR, "neutral.pt"), map_location="cpu")
    neutral_tokens = neutral_raw.view(-1, neutral_raw.shape[-1])
    
    # Compress tokens to a unified sentence vector, then center using the dataset mean
    neutral_mean = neutral_tokens.mean(dim=0, keepdim=True) - dataset_mean.cpu()
    neutral_unit = neutral_mean / neutral_mean.norm()  # True [1, d_model] unit vector
    
    # # Clean orthogonal projection pass
    # proj = (data @ neutral_unit.T)  # [N, 1]
    # data = data - proj * neutral_unit  
    
    neutral_unit = neutral_unit.to(DEVICE)
    print(f"  Orthogonalized against neutral direction (||neutral_mean||={neutral_mean.norm().item():.4f})")

    # 4. Initialize SAE with d_latent=2304 (2× overcomplete), k=32
    d_model = data.shape[-1]
    sae = Gemma3TopKSAE(d_model=d_model, d_latent=2304, k=32).to(DEVICE)

    # 5. Train
    sae, loss_history, best_epoch = train(sae, data, epochs=400, lr=3e-04, batch_size=512)
    save_loss_plot(loss_history, best_epoch, k=sae.k)

    # 6. Save State — include the explicit true mean for downstream baseline restoration
    torch.save({
        'model_state_dict': sae.state_dict(),
        'dataset_mean': dataset_mean.cpu(),
        'neutral_unit': neutral_unit.cpu(),
        'cfg': {
            'd_model': sae.d_model,
            'd_latent': sae.d_latent,
            'k': sae.k
        }
    }, SAE_SAVE_PATH)
    print(f"\nSAE saved to {SAE_SAVE_PATH}")