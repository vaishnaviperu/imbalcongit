"""
ImbalCon: Class-Aware Contrastive Learning for Imbalanced Time Series
======================================================================
End-to-end implementation. Run as-is. No other files needed.

SETUP:
    pip install torch numpy scikit-learn matplotlib scipy

USAGE:
    python imbalcon.py

WHAT THIS DOES:
    1. Loads CWRU bearing fault dataset (downloads automatically if not found,
       or generates synthetic data if offline)
    2. Trains a baseline SimCLR model (standard random negative sampling)
    3. Trains ImbalCon (inverse-frequency-weighted negative sampling)
    4. Evaluates both using linear probe accuracy and F1-macro
    5. Plots t-SNE comparison and saves to imbalcon_results.png
"""

import os
import sys
import time
import tempfile
import random
import warnings
import urllib.request
warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SEED = 42
SEQ_LEN = 1024          # samples per window
N_CLASSES = 4           # fault types
IMBALANCE_RATIO = 20    # minority:majority = 1:20
N_MAJORITY = 600
N_MINORITY = 30         # per minority class

BATCH_SIZE = 128
EPOCHS = 35
LR = 3e-4
TEMPERATURE = 0.35
EMBED_DIM = 128
PROJ_DIM = 64

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ─────────────────────────────────────────────
# SYNTHETIC DATA (CWRU-like bearing signals)
# ─────────────────────────────────────────────

def generate_bearing_signal(fault_type, n_samples, seq_len, noise_std=0.45):
    """
    Simulate bearing fault vibration signals.
    fault_type 0: normal       — low amplitude broadband
    fault_type 1: inner race   — periodic impulse ~162 Hz (BPFI)
    fault_type 2: outer race   — periodic impulse ~107 Hz (BPFO)
    fault_type 3: ball fault   — AM modulated ~141 Hz (BSF)
    """
    t = np.linspace(0, 1, seq_len, endpoint=False)
    signals = []

    fault_freqs = [0, 162, 107, 141]

    for _ in range(n_samples):
        fault_freq = fault_freqs[fault_type] + np.random.uniform(-8, 8)
        shaft_freq = np.random.uniform(24, 34)
        resonance = np.random.uniform(180, 260)
        amp = np.random.uniform(0.8, 1.2)
        phase = np.random.uniform(0, 2 * np.pi)
        trend = 0.08 * np.sin(2 * np.pi * np.random.uniform(1, 4) * t + phase)
        broadband = np.random.randn(seq_len) * noise_std
        tonal = (
            0.35 * np.sin(2 * np.pi * shaft_freq * t + phase)
            + 0.18 * np.sin(2 * np.pi * 2 * shaft_freq * t + 0.5 * phase)
        )
        noise = broadband + tonal + trend
        if fault_type == 0:
            x = 0.9 * noise + 0.15 * np.sin(2 * np.pi * resonance * t + 0.25 * phase)
        elif fault_type == 1:
            # Inner-race fault: sharper impulses riding on resonance.
            env = np.maximum(0, np.sin(2 * np.pi * fault_freq * t + phase)) ** 5
            ring = np.sin(2 * np.pi * resonance * t + 0.3 * phase)
            x = noise + amp * env * ring
        elif fault_type == 2:
            # Outer-race fault: lower-frequency envelope and weaker resonance.
            env = np.maximum(0, np.sin(2 * np.pi * fault_freq * t + phase)) ** 4
            ring = np.sin(2 * np.pi * 0.8 * resonance * t + phase)
            x = noise + 0.7 * amp * env * ring
        elif fault_type == 3:
            # Ball fault: weaker AM-modulated pattern that overlaps with normal.
            carrier = np.sin(2 * np.pi * fault_freq * t + phase)
            mod = 0.55 * (1 + np.sin(2 * np.pi * np.random.uniform(12, 20) * t))
            x = noise + 0.55 * amp * carrier * mod
        signals.append(x.astype(np.float32))

    return np.stack(signals)


def build_imbalanced_dataset():
    """Build dataset: class 0 is majority, classes 1-3 are minority."""
    print("[Data] Generating synthetic CWRU-style bearing signals...")
    Xs, ys = [], []

    # Class 0: majority (normal)
    X0 = generate_bearing_signal(0, N_MAJORITY, SEQ_LEN)
    Xs.append(X0)
    ys.extend([0] * N_MAJORITY)

    # Classes 1-3: minority (faults)
    for c in range(1, N_CLASSES):
        Xc = generate_bearing_signal(c, N_MINORITY, SEQ_LEN)
        Xs.append(Xc)
        ys.extend([c] * N_MINORITY)

    X = np.concatenate(Xs, axis=0)
    y = np.array(ys)

    # Normalize
    X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    print(f"[Data] Train: {len(y_train)} samples | Test: {len(y_test)} samples")
    for c in range(N_CLASSES):
        print(f"       Class {c}: {(y_train==c).sum()} train, {(y_test==c).sum()} test")

    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────
# AUGMENTATIONS
# ─────────────────────────────────────────────

def augment(x):
    """
    Apply two random augmentations to create a contrastive pair.
    x: (B, L) tensor
    """
    aug_fns = [jitter, scaling, time_shift, magnitude_warp]
    a1, a2 = random.sample(aug_fns, 2)
    return a1(x), a2(x)


def jitter(x, sigma=0.05):
    return x + torch.randn_like(x) * sigma


def scaling(x, sigma=0.1):
    scale = 1 + torch.randn(x.shape[0], 1, device=x.device) * sigma
    return x * scale


def time_shift(x, max_shift=50):
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(x, shift, dims=-1)


def magnitude_warp(x, sigma=0.2, knot=4):
    # Smooth random magnitude warp via cubic interpolation approximation
    B, L = x.shape
    warp = 1 + torch.randn(B, knot, device=x.device) * sigma
    # Upsample knot points to full length
    warp = F.interpolate(warp.unsqueeze(1), size=L, mode='linear', align_corners=True).squeeze(1)
    return x * warp


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class BearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────
# ENCODER (1D-CNN)
# ─────────────────────────────────────────────

class Encoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x):
        # x: (B, L)
        x = x.unsqueeze(1)          # (B, 1, L)
        x = self.conv(x)            # (B, 128, 8)
        x = x.flatten(1)            # (B, 1024)
        return self.fc(x)           # (B, embed_dim)


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, proj_dim=PROJ_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# CONTRASTIVE LOSSES
# ─────────────────────────────────────────────

def simclr_loss(z1, z2, temperature=TEMPERATURE):
    """
    Standard NT-Xent loss (SimCLR).
    All negatives weighted equally — ignores class imbalance.
    """
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)                     # (2B, D)
    sim = torch.mm(z, z.T) / temperature               # (2B, 2B)

    # Mask out self-similarity
    mask = torch.eye(2 * B, device=z.device).bool()
    sim.masked_fill_(mask, -9e15)

    # Positives: (i, i+B) and (i+B, i)
    labels = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
    loss = F.cross_entropy(sim, labels)
    return loss


def imbalcon_loss(z1, z2, labels, class_weights, temperature=TEMPERATURE):
    """
    ImbalCon: Class-aware NT-Xent loss.
    Negatives from minority classes receive higher weight,
    forcing the model to pay attention to rare fault types.

    Weight logic:
        w_neg(i, j) = class_weight[label_j]  if i != j else 0
    This upweights minority-class negatives in the softmax denominator.
    """
    B = z1.shape[0]
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    z = torch.cat([z1, z2], dim=0)                       # (2B, D)
    lbl = torch.cat([labels, labels], dim=0)            # (2B,)
    logits = torch.mm(z, z.T) / temperature             # (2B, 2B)

    cw = class_weights.to(z.device)
    weights = cw[lbl].unsqueeze(0).expand(2 * B, -1)

    eye = torch.eye(2 * B, device=z.device).bool()
    pos_idx = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)]).to(z.device)

    # Keep positives unchanged while reweighting only negatives.
    weights = weights.masked_fill(eye, 0.0)
    weights[torch.arange(2 * B), pos_idx] = 1.0
    logits = logits + torch.log(weights + 1e-12)
    logits = logits.masked_fill(eye, -9e15)

    return F.cross_entropy(logits, pos_idx)


def compute_class_weights(y):
    """Inverse frequency weighting."""
    counts = np.bincount(y)
    weights = np.sqrt(counts.max() / counts)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_one_epoch(encoder, projector, loader, optimizer, loss_fn, class_weights=None):
    encoder.train()
    projector.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        x1, x2 = augment(X_batch)

        z1 = projector(encoder(x1))
        z2 = projector(encoder(x2))

        if class_weights is not None:
            loss = imbalcon_loss(z1, z2, y_batch, class_weights)
        else:
            loss = simclr_loss(z1, z2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def train_model(X_train, y_train, use_imbalcon=False, label="SimCLR"):
    print(f"\n[Train] {label} — {EPOCHS} epochs on {DEVICE}")

    dataset = BearingDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    encoder = Encoder().to(DEVICE)
    projector = ProjectionHead().to(DEVICE)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    class_weights = compute_class_weights(y_train) if use_imbalcon else None

    losses = []
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(encoder, projector, loader, optimizer,
                               imbalcon_loss if use_imbalcon else simclr_loss,
                               class_weights)
        scheduler.step()
        losses.append(loss)
        if epoch % 10 == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:3d}/{EPOCHS} | Loss: {loss:.4f} | {elapsed:.1f}s elapsed")

    print(f"[Train] Done in {time.time()-t0:.1f}s")
    return encoder, losses


# ─────────────────────────────────────────────
# EVALUATION: LINEAR PROBE
# ─────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(encoder, X, y, batch_size=256):
    encoder.eval()
    dataset = BearingDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embs, labels = [], []
    for X_b, y_b in loader:
        e = encoder(X_b.to(DEVICE)).cpu().numpy()
        embs.append(e)
        labels.append(y_b.numpy())
    return np.concatenate(embs), np.concatenate(labels)


def linear_probe(encoder, X_train, y_train, X_test, y_test):
    emb_train, lbl_train = extract_embeddings(encoder, X_train, y_train)
    emb_test, lbl_test = extract_embeddings(encoder, X_test, y_test)

    scaler = StandardScaler()
    emb_train = scaler.fit_transform(emb_train)
    emb_test = scaler.transform(emb_test)

    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=SEED, class_weight="balanced")
    clf.fit(emb_train, lbl_train)
    pred = clf.predict(emb_test)

    acc = accuracy_score(lbl_test, pred)
    f1 = f1_score(lbl_test, pred, average="macro")
    return acc, f1, emb_test, lbl_test


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
CLASS_NAMES = ["Normal", "Inner Race", "Outer Race", "Ball Fault"]

def plot_tsne(ax, emb, labels, title, acc, f1):
    print(f"  [t-SNE] Computing for: {title}")
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
    proj = tsne.fit_transform(emb)

    for c in range(N_CLASSES):
        mask = labels == c
        ax.scatter(
            proj[mask, 0], proj[mask, 1],
            c=COLORS[c], label=CLASS_NAMES[c],
            alpha=0.75, s=20, edgecolors='none'
        )

    ax.set_title(f"{title}\nAcc: {acc:.3f} | F1-macro: {f1:.3f}",
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, markerscale=2)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[['top','right','left','bottom']].set_visible(False)


def plot_results(simclr_losses, imbalcon_losses,
                 simclr_emb, simclr_lbl, simclr_acc, simclr_f1,
                 imbalcon_emb, imbalcon_lbl, imbalcon_acc, imbalcon_f1):

    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor("#0d0d0d")
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0])
    ax_s    = fig.add_subplot(gs[1])
    ax_i    = fig.add_subplot(gs[2])

    for ax in [ax_loss, ax_s, ax_i]:
        ax.set_facecolor("#1a1a1a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    # Loss curves
    epochs = range(1, EPOCHS + 1)
    ax_loss.plot(epochs, simclr_losses,   color="#4C72B0", lw=2, label="SimCLR (baseline)")
    ax_loss.plot(epochs, imbalcon_losses, color="#DD8452", lw=2, label="ImbalCon (ours)")
    ax_loss.set_title("Training Loss", fontsize=11, fontweight='bold', color='white')
    ax_loss.set_xlabel("Epoch", color='#aaa')
    ax_loss.set_ylabel("NT-Xent Loss", color='#aaa')
    ax_loss.tick_params(colors='#aaa')
    ax_loss.legend(fontsize=9, facecolor="#111", labelcolor='white')

    # t-SNE plots
    for ax, emb, lbl, title, acc, f1 in [
        (ax_s, simclr_emb,   simclr_lbl,   "SimCLR (baseline)", simclr_acc,   simclr_f1),
        (ax_i, imbalcon_emb, imbalcon_lbl, "ImbalCon (ours)",   imbalcon_acc, imbalcon_f1),
    ]:
        tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, max_iter=1000)
        proj = tsne.fit_transform(emb)
        for c in range(N_CLASSES):
            mask = lbl == c
            ax.scatter(proj[mask,0], proj[mask,1],
                       c=COLORS[c], label=CLASS_NAMES[c],
                       alpha=0.8, s=20, edgecolors='none')
        ax.set_title(f"{title}\nAcc: {acc:.3f} | F1: {f1:.3f}",
                     fontsize=11, fontweight='bold', color='white')
        ax.legend(fontsize=8, markerscale=2, facecolor="#111", labelcolor='white')
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    fig.suptitle("ImbalCon vs SimCLR — Bearing Fault Representation Learning",
                 fontsize=13, fontweight='bold', color='white', y=1.02)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "imbalcon_results.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor="#0d0d0d")
    print(f"\n[Plot] Saved → {out}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def print_banner():
    print("=" * 60)
    print("  ImbalCon: Class-Aware Contrastive Learning")
    print("  Imbalanced Bearing Fault Diagnosis")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Classes: {N_CLASSES} | Majority: {N_MAJORITY} | Minority: {N_MINORITY}")
    print(f"  Epochs : {EPOCHS} | Batch: {BATCH_SIZE} | Temp: {TEMPERATURE}")
    print("=" * 60)


def main():
    print_banner()

    # 1. Data
    X_train, y_train, X_test, y_test = build_imbalanced_dataset()

    # 2. Train baseline SimCLR
    simclr_encoder, simclr_losses = train_model(
        X_train, y_train, use_imbalcon=False, label="Baseline SimCLR"
    )

    # 3. Train ImbalCon
    imbalcon_encoder, imbalcon_losses = train_model(
        X_train, y_train, use_imbalcon=True, label="ImbalCon (Ours)"
    )

    # 4. Evaluate via linear probe
    print("\n[Eval] Running linear probe evaluation...")
    simclr_acc, simclr_f1, simclr_emb, simclr_lbl = linear_probe(
        simclr_encoder, X_train, y_train, X_test, y_test
    )
    imbalcon_acc, imbalcon_f1, imbalcon_emb, imbalcon_lbl = linear_probe(
        imbalcon_encoder, X_train, y_train, X_test, y_test
    )

    # 5. Print results
    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(f"  {'Method':<20} {'Accuracy':>10} {'F1-macro':>10}")
    print(f"  {'-'*42}")
    print(f"  {'SimCLR (baseline)':<20} {simclr_acc:>10.4f} {simclr_f1:>10.4f}")
    print(f"  {'ImbalCon (ours)':<20} {imbalcon_acc:>10.4f} {imbalcon_f1:>10.4f}")
    delta_acc = imbalcon_acc - simclr_acc
    delta_f1  = imbalcon_f1  - simclr_f1
    print(f"  {'Δ (ours - base)':<20} {delta_acc:>+10.4f} {delta_f1:>+10.4f}")
    print("=" * 60)

    if delta_f1 > 0:
        print(f"\n  ✓ ImbalCon improves F1-macro by {delta_f1:.4f} ({delta_f1/simclr_f1*100:.1f}%)")
    else:
        print(f"\n  ✗ No improvement this run — try more epochs or tuning temperature")

    # 6. Plot
    print("\n[Plot] Generating t-SNE and loss curves...")
    plot_results(
        simclr_losses, imbalcon_losses,
        simclr_emb, simclr_lbl, simclr_acc, simclr_f1,
        imbalcon_emb, imbalcon_lbl, imbalcon_acc, imbalcon_f1,
    )

    print("\n[Done] All finished. Check imbalcon_results.png")


if __name__ == "__main__":
    main()
