"""Shared code for the optimizer loss landscape study (Steps 1 to 6)."""
import os, re, json, time, copy, platform, random, subprocess
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from lion_pytorch import Lion

# ---------------- paths and config ----------------
REPO_NAME = "optimizer-loss-landscape-study"
GITHUB_USER = "uddipanr"
REPO_DIR = f"/kaggle/working/{REPO_NAME}"
CKPT_DIR = f"{REPO_DIR}/checkpoints"
LOG_DIR = f"{REPO_DIR}/logs"
CFG_DIR = f"{REPO_DIR}/configs"
FIG_DIR = f"{REPO_DIR}/figures"
DATA_DIR = "/kaggle/working/data"
for _d in [CKPT_DIR, LOG_DIR, CFG_DIR, FIG_DIR, DATA_DIR]:
    os.makedirs(_d, exist_ok=True)

# Keep heavy weight files out of GitHub
_gi = f"{REPO_DIR}/.gitignore"
_gi_text = open(_gi).read() if os.path.exists(_gi) else ""
if "checkpoints/" not in _gi_text:
    with open(_gi, "a") as _fh:
        _fh.write("\ncheckpoints/\n")

CONFIG = {
    "dataset": "CIFAR10", "architecture": "resnet18", "epochs": 50,
    "batch_size": 128, "num_workers": 2,
    "seeds": [42, 7, 123], "optimizers": ["sgd", "adam", "adamw", "lion"],
}

# ---------------- reproducibility ----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# ---------------- data ----------------
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
test_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=test_transform)
eval_train_dataset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=False, transform=test_transform)
eval_train_loader = DataLoader(eval_train_dataset, batch_size=256, shuffle=False, num_workers=2)

def get_dataloaders(seed=42, batch_size=None):
    batch_size = batch_size or CONFIG["batch_size"]
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=CONFIG["num_workers"], generator=g)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=CONFIG["num_workers"])
    return train_loader, test_loader

# ---------------- model ----------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, 1)
        self.layer2 = self._make_layer(128, 2, 2)
        self.layer3 = self._make_layer(256, 2, 2)
        self.layer4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_channels, out_channels, s))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = torch.flatten(self.avgpool(out), 1)
        return self.fc(out)

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        nn.init.constant_(m.bias, 0)

def build_model(seed=42, device="cuda"):
    set_seed(seed)
    model = ResNet18(num_classes=10)
    model.apply(init_weights)
    return model.to(device)

# ---------------- optimizers ----------------
OPTIMIZER_CONFIGS = {
    "sgd":   {"lr": 0.1, "momentum": 0.9, "dampening": 0, "nesterov": False, "weight_decay": 5e-4},
    "adam":  {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 5e-4, "amsgrad": False},
    "adamw": {"lr": 1e-3, "betas": (0.9, 0.999), "eps": 1e-8, "weight_decay": 5e-2, "amsgrad": False},
    "lion":  {"lr": 1e-4, "betas": (0.9, 0.99), "weight_decay": 1e-1},
}
# Apply the tuned learning rates from the Step 5 sweep
_best = f"{CFG_DIR}/best_lrs.json"
if os.path.exists(_best):
    with open(_best) as _f:
        for _n, _lr in json.load(_f).items():
            OPTIMIZER_CONFIGS[_n]["lr"] = _lr

WEIGHT_DECAY_TYPE = {
    "sgd": "L2 penalty added to gradient (coupled)",
    "adam": "L2 penalty added to gradient (coupled, distorted by adaptive scaling)",
    "adamw": "decoupled, applied directly to weights",
    "lion": "decoupled, applied directly to weights",
}
OPTIMIZER_CLASS_NAME = {"sgd": "torch.optim.SGD", "adam": "torch.optim.Adam",
                        "adamw": "torch.optim.AdamW", "lion": "lion_pytorch.Lion"}
STATE_KEYS = {"sgd": ["momentum_buffer"], "adam": ["exp_avg", "exp_avg_sq"],
              "adamw": ["exp_avg", "exp_avg_sq"], "lion": ["exp_avg"]}

def build_optimizer(name, model, overrides=None):
    hp = copy.deepcopy(OPTIMIZER_CONFIGS[name])
    if overrides:
        hp.update(overrides)
    params = model.parameters()
    if name == "sgd":
        opt = torch.optim.SGD(params, **hp)
    elif name == "adam":
        opt = torch.optim.Adam(params, **hp)
    elif name == "adamw":
        opt = torch.optim.AdamW(params, **hp)
    elif name == "lion":
        opt = Lion(params, **hp)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    return opt, hp

def print_optimizer_summary(name, hp, optimizer):
    print("=" * 70)
    print(f"OPTIMIZER SUMMARY: {name.upper()}")
    print("=" * 70)
    print(f"Class             : {OPTIMIZER_CLASS_NAME[name]}")
    print(f"Weight decay type : {WEIGHT_DECAY_TYPE[name]}")
    print("Hyperparameters   :")
    for k, v in hp.items():
        print(f"    {k:<14}: {v}")
    print("Full param_group settings (as seen inside the optimizer):")
    for k, v in optimizer.param_groups[0].items():
        if k != "params":
            print(f"    {k:<14}: {v}")
    print("=" * 70)

# ---------------- helpers ----------------
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, 100.0 * correct / n

def global_grad_norm(model):
    norms = [p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
    return torch.norm(torch.stack(norms), 2)

@torch.no_grad()
def global_weight_norm(model):
    return torch.norm(torch.stack([p.detach().norm(2) for p in model.parameters()]), 2).item()

@torch.no_grad()
def get_flat_params(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])

@torch.no_grad()
def optimizer_state_stats(optimizer, name):
    stats = {}
    for key in STATE_KEYS[name]:
        norms = []
        for group in optimizer.param_groups:
            for p in group["params"]:
                st = optimizer.state.get(p, {})
                if key in st and st[key] is not None:
                    norms.append(st[key].detach().norm(2))
        stats[f"opt_{key}_norm"] = torch.norm(torch.stack(norms), 2).item() if norms else float("nan")
    return stats

# ---------------- training ----------------
def train_run(optimizer_name, seed=42, epochs=None, overrides=None,
              snapshot_every=5, tag="", skip_if_done=True):
    epochs = epochs or CONFIG["epochs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seed(seed)
    model = build_model(seed=seed, device=device)
    train_loader, test_loader = get_dataloaders(seed=seed)
    optimizer, hp = build_optimizer(optimizer_name, model, overrides)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    run_name = f"{optimizer_name}_{seed}_{hp['lr']}" + (f"_{tag}" if tag else "")
    run_ckpt_dir = f"{CKPT_DIR}/{run_name}"
    if skip_if_done and os.path.exists(f"{run_ckpt_dir}/final_model.pt"):
        print(f"Run {run_name} already finished. Skipping.")
        return run_name, None, None
    os.makedirs(run_ckpt_dir, exist_ok=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nRUN: {run_name}")
    print(f"Seed: {seed} | Epochs: {epochs} | Batch size: {CONFIG['batch_size']} | "
          f"Device: {device} | Parameters: {total_params:,}")
    print(f"Scheduler: CosineAnnealingLR (T_max={epochs}) | Loss: CrossEntropyLoss | "
          f"Mixed precision: off | Grad clipping: none")
    print_optimizer_summary(optimizer_name, hp, optimizer)

    init_flat = get_flat_params(model).clone()
    torch.save(init_flat.half().cpu(), f"{run_ckpt_dir}/snap_epoch_000.pt")

    history, diverged, start_time = [], False, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        flat_before = get_flat_params(model)
        run_loss = torch.zeros((), device=device)
        run_correct = torch.zeros((), device=device)
        grad_norm_sum = torch.zeros((), device=device)
        n_samples, n_steps = 0, 0

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            grad_norm_sum += global_grad_norm(model)
            optimizer.step()
            run_loss += loss.detach() * x.size(0)
            run_correct += (out.argmax(1) == y).sum()
            n_samples += x.size(0)
            n_steps += 1

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()

        train_loss = run_loss.item() / n_samples
        train_acc = 100.0 * run_correct.item() / n_samples
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        flat_after = get_flat_params(model)
        weight_change = (flat_after - flat_before).norm().item()

        row = {
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "test_loss": test_loss, "test_acc": test_acc,
            "gen_gap_running": train_acc - test_acc,
            "lr": current_lr,
            "avg_grad_norm": grad_norm_sum.item() / n_steps,
            "weight_norm": global_weight_norm(model),
            "epoch_weight_change": weight_change,
            "update_to_weight_ratio": weight_change / (flat_before.norm().item() + 1e-12),
            "dist_from_init": (flat_after - init_flat).norm().item(),
            "epoch_time_sec": time.time() - t0,
        }
        row.update(optimizer_state_stats(optimizer, optimizer_name))
        history.append(row)

        print(f"Ep {epoch:>3}/{epochs} | lr {current_lr:.2e} | "
              f"train loss {train_loss:.4f} acc {train_acc:.2f} | "
              f"test loss {test_loss:.4f} acc {test_acc:.2f} | "
              f"grad norm {row['avg_grad_norm']:.3f} | "
              f"w norm {row['weight_norm']:.1f} | {row['epoch_time_sec']:.0f}s")

        if epoch % snapshot_every == 0 or epoch == epochs:
            torch.save(flat_after.half().cpu(), f"{run_ckpt_dir}/snap_epoch_{epoch:03d}.pt")
        if not np.isfinite(train_loss):
            print("Training diverged (loss is NaN or infinite). Stopping this run.")
            diverged = True
            break

    total_time = time.time() - start_time
    final_train_loss, final_train_acc = evaluate(model, eval_train_loader, criterion, device)
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    final_metrics = {
        "final_train_loss": final_train_loss, "final_train_acc": final_train_acc,
        "final_test_loss": final_test_loss, "final_test_acc": final_test_acc,
        "generalization_gap": final_train_acc - final_test_acc,
        "diverged": diverged, "total_train_time_sec": total_time,
    }

    torch.save({
        "model_state_dict": model.state_dict(), "run_name": run_name,
        "optimizer_name": optimizer_name, "hyperparameters": hp, "seed": seed,
        "epochs_trained": len(history), "final_metrics": final_metrics,
    }, f"{run_ckpt_dir}/final_model.pt")

    pd.DataFrame(history).to_csv(f"{LOG_DIR}/{run_name}_history.csv", index=False)

    run_config = {
        "run_name": run_name, "timestamp": datetime.now().isoformat(timespec="seconds"),
        "seed": seed, "dataset": CONFIG["dataset"], "architecture": CONFIG["architecture"],
        "total_parameters": total_params, "epochs": epochs,
        "batch_size": CONFIG["batch_size"], "num_workers": CONFIG["num_workers"],
        "train_augmentation": "RandomCrop(32, padding=4) + RandomHorizontalFlip",
        "normalization_mean": CIFAR10_MEAN, "normalization_std": CIFAR10_STD,
        "loss_function": "CrossEntropyLoss",
        "optimizer": {"name": optimizer_name, "class": OPTIMIZER_CLASS_NAME[optimizer_name],
                      "weight_decay_type": WEIGHT_DECAY_TYPE[optimizer_name],
                      "hyperparameters": hp},
        "scheduler": {"type": "CosineAnnealingLR", "T_max": epochs, "final_lr": 0.0},
        "mixed_precision": False, "gradient_clipping": None,
        "snapshot_every_n_epochs": snapshot_every,
        "weight_init": "Kaiming normal (conv), constant (BN), N(0, 0.01) (linear)",
        "environment": {"device": device,
                        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "none",
                        "torch_version": torch.__version__,
                        "python_version": platform.python_version()},
        "results": final_metrics,
    }
    with open(f"{CFG_DIR}/{run_name}_config.json", "w") as f:
        json.dump(run_config, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print(f"RUN COMPLETE: {run_name}")
    print("=" * 70)
    print(f"Final train accuracy (clean) : {final_train_acc:.2f}%")
    print(f"Final test accuracy          : {final_test_acc:.2f}%")
    print(f"Generalization gap           : {final_metrics['generalization_gap']:.2f}")
    print(f"Final train / test loss      : {final_train_loss:.4f} / {final_test_loss:.4f}")
    print(f"Total training time          : {total_time/60:.1f} min")
    print(f"Saved: {run_ckpt_dir}/final_model.pt, history CSV, config JSON, snapshots")
    print("=" * 70)
    return run_name, history, final_metrics

# ---------------- plotting and loading ----------------
def plot_run(run_name):
    df = pd.read_csv(f"{LOG_DIR}/{run_name}_history.csv")
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle(run_name, fontsize=14)
    axes[0, 0].plot(df.epoch, df.train_loss, label="train")
    axes[0, 0].plot(df.epoch, df.test_loss, label="test")
    axes[0, 0].set_title("Loss"); axes[0, 0].legend()
    axes[0, 1].plot(df.epoch, df.train_acc, label="train")
    axes[0, 1].plot(df.epoch, df.test_acc, label="test")
    axes[0, 1].set_title("Accuracy (%)"); axes[0, 1].legend()
    axes[0, 2].plot(df.epoch, df.lr); axes[0, 2].set_title("Learning rate")
    axes[1, 0].plot(df.epoch, df.avg_grad_norm); axes[1, 0].set_title("Average gradient norm")
    axes[1, 1].plot(df.epoch, df.weight_norm); axes[1, 1].set_title("Weight norm")
    axes[1, 2].plot(df.epoch, df.dist_from_init); axes[1, 2].set_title("Distance from initial weights")
    for ax in axes.flat:
        ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{run_name}_curves.png", dpi=150)
    plt.close(fig)

def load_trained_model(run_name, device="cuda"):
    ckpt = torch.load(f"{CKPT_DIR}/{run_name}/final_model.pt", map_location=device, weights_only=False)
    model = ResNet18(num_classes=10).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt

# ---------------- git ----------------
def _clean(text):
    """Hide any token that appears inside a URL before printing."""
    return re.sub(r"https://[^@\s]+@", "https://***@", text or "")

def _ensure_git_identity():
    have = subprocess.run(["git", "config", "user.email"], cwd=REPO_DIR,
                          capture_output=True, text=True).stdout.strip()
    if not have:
        subprocess.run(["git", "config", "user.email", f"{GITHUB_USER}@users.noreply.github.com"], cwd=REPO_DIR)
        subprocess.run(["git", "config", "user.name", "Uddipan"], cwd=REPO_DIR)

def _refresh_remote():
    try:
        from kaggle_secrets import UserSecretsClient
        token = UserSecretsClient().get_secret("GITHUB_TOKEN")
        subprocess.run(["git", "remote", "set-url", "origin",
                        f"https://{token}@github.com/{GITHUB_USER}/{REPO_NAME}.git"],
                       cwd=REPO_DIR, capture_output=True)
    except Exception as e:
        print("Could not refresh git remote:", type(e).__name__)

def push_to_github(commit_message):
    _ensure_git_identity()
    _refresh_remote()
    def git(*args):
        return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True, text=True)
    git("add", ".")
    c = git("commit", "-m", commit_message)
    if "nothing to commit" in (c.stdout + c.stderr):
        print("Nothing new to push.")
        return
    git("pull", "--rebase")
    p = git("push")
    if p.returncode == 0:
        print("Pushed:", commit_message)
    else:
        print("PUSH FAILED:", _clean(p.stderr)[-300:])

def final_run_name(name, seed):
    return f"{name}_{seed}_{OPTIMIZER_CONFIGS[name]['lr']}"

def run_seed(seed, optimizers=("sgd", "adam", "adamw", "lion")):
    for name in optimizers:
        print("\n" + "#" * 70)
        print(f"# SEED {seed} | OPTIMIZER {name.upper()} | lr {OPTIMIZER_CONFIGS[name]['lr']}")
        print("#" * 70)
        run_name, _, _ = train_run(name, seed=seed, epochs=CONFIG["epochs"], snapshot_every=5)
        if os.path.exists(f"{LOG_DIR}/{run_name}_history.csv"):
            plot_run(run_name)
        push_to_github(f"Step 6: seed {seed}, {name} ({run_name})")
    print(f"\nSeed {seed} finished.")
