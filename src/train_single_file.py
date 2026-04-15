#!/usr/bin/env python

import os
import compress_pickle as pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import datetime
import random
import sys
import shutil
import time

# ── Config ────────────────────────────────────────────────────────────────────

rootDir    = '/home/owensr/chess'
lr         = 0.0003
batch_size = 256

# ── Helpers ───────────────────────────────────────────────────────────────────

def ts():
    return datetime.datetime.now().strftime("%H:%M:%S")

# ── Network ───────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.action_size = 8 * 8 * 73
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(256)

    def forward(self, s):
        s = s.view(-1, 22, 8, 8)
        s = F.relu(self.bn1(self.conv1(s)))
        return s


class ResBlock(nn.Module):
    def __init__(self, inplanes=256, planes=256, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class OutBlock(nn.Module):
    def __init__(self):
        super().__init__()
        # value head
        self.conv  = nn.Conv2d(256, 1, kernel_size=1)
        self.bn    = nn.BatchNorm2d(1)
        self.fc1   = nn.Linear(8 * 8, 64)
        self.fc2   = nn.Linear(64, 1)
        # policy head
        self.conv1      = nn.Conv2d(256, 128, kernel_size=1)
        self.bn1        = nn.BatchNorm2d(128)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.fc         = nn.Linear(8 * 8 * 128, 8 * 8 * 73)

    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))
        v = v.view(-1, 8 * 8)
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))
        p = p.view(-1, 8 * 8 * 128)
        p = self.logsoftmax(self.fc(p)).exp()
        return p, v


class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()

    def forward(self, s):
        s = self.conv(s)
        for block in range(19):
            s = getattr(self, "res_%i" % block)(s)
        return self.outblock(s)


class AlphaLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_value, value, y_policy, policy):
        value_error  = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-6 + y_policy.float()).float().log()), 1)
        return (value_error.view(-1).float() + policy_error).mean()

# ── Data loading ──────────────────────────────────────────────────────────────

class board_data(Dataset):
    def __init__(self, dataset):  # dataset = np.array of (s, p, v)
        self.X        = dataset[:, 0]
        self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx].transpose(2, 0, 1), self.y_p[idx], self.y_v[idx]


class board_data_all(IterableDataset):
    def __init__(self, directory, runtime):
        super().__init__()
        self.runtime   = runtime
        self.starttime = time.time()
        self.files     = [os.path.join(directory, f) for f in os.listdir(directory)]
        self.loaders   = []

    def generate(self):
        files_loaded   = 0
        records_yielded = 0
        last_log       = time.time()
        print(f"[{ts()}][loader] starting — {len(self.files)} files available, runtime {self.runtime}s", flush=True)
        while (len(self.files) > 0 or len(self.loaders) > 0) and (time.time() < self.starttime + self.runtime):
            data_item = None
            while data_item is None and not (len(self.files) == 0 and len(self.loaders) == 0) and (time.time() < self.starttime + self.runtime):
                if len(self.loaders) < 2 and len(self.files) > 0:
                    file = random.choice(self.files)
                    self.files.remove(file)
                    files_loaded += 1
                    print(f"[{ts()}][loader] loading file {files_loaded} — {os.path.basename(file)} ({len(self.files)} remaining, {len(self.loaders)} active)", flush=True)
                    with open(file, 'rb') as fo:
                        try:
                            data = pickle.load(fo)
                        except EOFError:
                            print(f"[{ts()}][loader] EOFError in {file}", flush=True)
                            data = []
                    data = np.array(data, dtype="object")
                    print(f"[{ts()}][loader] file loaded — {len(data)} records", flush=True)
                    self.loaders.append(iter(DataLoader(board_data(data), shuffle=False, pin_memory=False)))
                loader    = random.choice(self.loaders)
                data_item = next(loader, None)
                if data_item is None:
                    self.loaders.remove(loader)
            if data_item is not None:
                records_yielded += 1
                now = time.time()
                if now - last_log >= 10:
                    elapsed = now - self.starttime
                    print(f"[{ts()}][loader] {records_yielded:,} records yielded | elapsed {elapsed:.0f}s/{self.runtime}s | {len(self.loaders)} active loaders", flush=True)
                    last_log = now
                yield (torch.squeeze(data_item[0]),
                       torch.squeeze(data_item[1]),
                       torch.squeeze(data_item[2]))

    def __iter__(self):
        return iter(self.generate())

# ── Training loop ─────────────────────────────────────────────────────────────

def train(net, train_path, lr, batch_size, run, runtime, device):
    cuda      = device.type == "cuda"
    criterion = AlphaLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=lr)
    net.train()

    roll_99 = 7.0
    roll_9  = 7.0
    total_loss       = 0.0
    losses_per_batch = []

    train_loader = DataLoader(
        board_data_all(train_path, runtime),
        batch_size=batch_size, num_workers=1, pin_memory=cuda
    )

    train_start = time.time()
    batch_start = time.time()
    print(f"[{ts()}][train] waiting for first batch (batch_size={batch_size})...", flush=True)

    for i, data in enumerate(train_loader, 0):
        batch_load_time = time.time() - batch_start
        state, policy, value = data
        state  = state.to(device, dtype=torch.float32)
        policy = policy.to(device, dtype=torch.float32)
        value  = value.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        policy_pred, value_pred = net(state)
        loss = criterion(value_pred[:, 0], value, policy_pred, policy)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        roll_99 = roll_99 * 0.99 + loss.item() * 0.01
        roll_9  = roll_9  * 0.9  + loss.item() * 0.1
        elapsed = time.time() - train_start
        print(f"[{ts()}][train] batch:{i} loss:{loss.item():.4f} l9:{roll_9:.4f} l99:{roll_99:.4f} | load:{batch_load_time:.1f}s elapsed:{elapsed:.0f}s", flush=True)
        batch_start = time.time()
        losses_per_batch.append(loss.item())

        if i % 100 == 99:
            print(f"[{ts()}][train] --- checkpoint: {i+1} batches, avg loss {total_loss/(i+1):.4f}, run={run}, lr={lr}", flush=True)
            print(f"  Policy (target/pred): {policy[0].argmax().item()} / {policy_pred[0].argmax().item()}")
            print(f"  Value  (target/pred): {value[0].item():.4f} / {value_pred[0,0].item():.4f}")

    return roll_99

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: train_single_file.py <runId> <trainDir> <runtimeSeconds>")
        sys.exit(1)

    run      = int(sys.argv[1])
    trainDir = sys.argv[2]
    runtime  = int(sys.argv[3])

    mp.set_start_method("spawn", force=True)

    train_path = os.path.join(rootDir, "data", trainDir)

    cuda   = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print(f"[{ts()}] Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if cuda else ""), flush=True)

    net = ChessNet().to(device)

    model_path = os.path.join(rootDir, "data", "model_data", "latest.gz")
    print(f"[{ts()}] Loading model from {model_path}", flush=True)
    checkpoint    = torch.load(model_path, weights_only=True, map_location=device)
    remove_prefix = '_orig_mod.'
    state_dict    = {(k[len(remove_prefix):] if k.startswith(remove_prefix) else k): v
                     for k, v in checkpoint['state_dict'].items()}
    net.load_state_dict(state_dict)

    if cuda:
        print(f"[{ts()}] Compiling model with torch.compile()...", flush=True)
        net = torch.compile(net)
        print(f"[{ts()}] Model compiled.", flush=True)

    net.share_memory()
    print(f"[{ts()}] Model loaded, starting training run={run} trainDir={trainDir} runtime={runtime}s lr={lr} batch_size={batch_size}", flush=True)

    loss_99  = train(net, train_path, lr, batch_size, run, runtime, device)

    out_file = os.path.join(rootDir, "data", "model_data",
                            f"model_{run}_{trainDir}_{loss_99:.2f}_{datetime.datetime.today().strftime('%Y-%m-%d-%H%M%S')}.gz")
    torch.save({'state_dict': net.state_dict()}, out_file)
    shutil.copy(out_file, os.path.join(rootDir, "data", "model_data", "latest.gz"))
    print(f"[{ts()}] Saved model to {out_file} and updated latest.gz", flush=True)
