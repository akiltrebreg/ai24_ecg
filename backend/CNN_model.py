import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, AUROC, Precision, Recall, F1Score

from typing import Tuple, Optional
from logger_config import logger

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ECG1DCNN(nn.Module):
    def __init__(self, n_leads: int = 12, num_classes: int = 30):
        super().__init__()
        # Начальный сверточный блок
        self.conv0 = nn.Conv1d(n_leads, 64, kernel_size=7, padding=3)
        self.bn0 = nn.BatchNorm1d(64)
        self.pool0 = nn.MaxPool1d(kernel_size=2)    # T → T/2

        # Residual-блоки с расширением каналов
        self.block1 = nn.Sequential(
            ResidualBlock(64, 7),
            nn.MaxPool1d(2),                           # T/2 → T/4
            nn.Conv1d(64, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            ResidualBlock(128, 5),
            nn.MaxPool1d(2),                           # T/4 → T/8
            nn.Conv1d(128, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.block3 = nn.Sequential(
            ResidualBlock(256, 3),
            nn.MaxPool1d(2),                           # T/8 → T/16
            nn.Conv1d(256, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
        )
        self.block4 = nn.Sequential(
            ResidualBlock(512, 3),
            nn.MaxPool1d(2),                           # T/16 → T/32
        )

        # Глобальный пул + классификатор
        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.pool0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.global_pool(x)          # (batch, 512, 1)
        x = x.view(x.size(0), -1)        # (batch, 512)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)             # (batch, num_classes)
        return logits


class ECGDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int):
        self.signals = df['signal'].tolist()
        self.labels = df['labels_target'].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        mat = np.array(self.signals[idx], dtype=np.float32)  # (12, T)
        T = mat.shape[1]
        if T < self.max_len:
            pad = np.zeros((12, self.max_len - T), dtype=np.float32)
            mat = np.concatenate([mat, pad], axis=1)
        elif T > self.max_len:
            mat = mat[:, :self.max_len]
        X = torch.from_numpy(mat)              # (12, max_len)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return X, y

class ECGInferenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int):
        # сигнал остаётся прежним
        self.signals = df['signal'].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        mat = np.array(self.signals[idx], dtype=np.float32)  # (12, T)
        T = mat.shape[1]
        if T < self.max_len:
            pad = np.zeros((12, self.max_len - T), dtype=np.float32)
            mat = np.concatenate([mat, pad], axis=1)
        elif T > self.max_len:
            mat = mat[:, :self.max_len]
        X = torch.from_numpy(mat)  # (12, max_len)
        return X



def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def validate_epoch(model, loader, criterion, device, average='weighted'):
    model.eval()
    total_loss = 0.0
    num_labels = loader.dataset[0][1].shape[0]
    acc = Accuracy(task="multilabel", num_labels=num_labels, average=average).to(device)
    auc = AUROC(task="multilabel", num_labels=num_labels, average=average).to(device)
    prec = Precision(task="multilabel", num_labels=num_labels, average=average).to(device)
    rec = Recall(task="multilabel", num_labels=num_labels, average=average).to(device)
    f1 = F1Score(task="multilabel", num_labels=num_labels, average=average).to(device)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * x.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            acc.update(preds, y.int())
            prec.update(preds, y.int())
            rec.update(preds, y.int())
            f1.update(preds, y.int())
            auc.update(probs, y.int())

    return (
        total_loss / len(loader.dataset),
        acc.compute().item(),
        auc.compute().item(),
        prec.compute().item(),
        rec.compute().item(),
        f1.compute().item()
    )


def train_model(model, train_loader, val_loader, device,
                epochs=25, lr=1e-4, pos_weight=None,
                save_path="", average='weighted', path_for_hp = ""):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_f1 = 0.0
    history = {k: [] for k in ['train_loss','val_loss','acc','auc','prec','rec','f1']}
    with open(path_for_hp, "w") as fp:
        json.dump({"epochs": epochs, "lr": lr, "average": average}, fp)
    for epoch in range(1, epochs+1):
        history['train_loss'].append(train_epoch(model, train_loader, criterion, optimizer, device))
        vals = validate_epoch(model, val_loader, criterion, device, average)
        history['val_loss'].append(vals[0])
        for k,v in zip(['acc','auc','prec','rec','f1'], vals[1:]):
            history[k].append(v)

        if history['f1'][-1] > best_f1:
            best_f1 = history['f1'][-1]
            torch.save(model.state_dict(), save_path)
        logger.info(f"Epoch {epoch}/{epochs}")

    print(f"Training finished. Best F1: {best_f1:.4f}")


def test_model(model, test_loader, device, checkpoint="best_ecg1dcnn.pth", average='weighted'):
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    res = validate_epoch(model, test_loader, criterion, device, average)
    result = {}
    for name, val in zip(['Loss','Acc','AUC','Prec','Rec','F1'], res):
        result[name] = val
    return result


def prepare_loaders(df, batch_size=32, test_size=0.2, val_size=0.5, random_state=42):
    train_df, tmp = train_test_split(df, test_size=test_size, random_state=random_state)
    val_df, test_df = train_test_split(tmp, test_size=val_size, random_state=random_state)
    max_len = max(len(sig[0]) for sig in train_df['signal'])
    train_loader = DataLoader(ECGDataset(train_df, max_len), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ECGDataset(val_df,   max_len), batch_size=batch_size)
    test_loader = DataLoader(ECGDataset(test_df,  max_len), batch_size=batch_size)
    labels = torch.tensor(train_df['labels_target'].tolist(), dtype=torch.float32)
    pos = labels.sum(dim=0)
    neg = labels.size(0) - pos
    pos_weight = neg / (pos + 1e-6)
    return train_loader, val_loader, test_loader, pos_weight




def predict(model: torch.nn.Module, test_loader: DataLoader, checkpoint: str,
            device: Optional[torch.device] = None, threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Загружает веса из checkpoint в модель и выполняет предсказание для всех данных из test_loader.

    Args:
        model: экземпляр nn.Module (ваша ECG1DCNN).
        test_loader: DataLoader для тестового набора.
        checkpoint: путь к файлу с сохранёнными весами (например, "best_ecg1dcnn.pth").
        device: torch.device, на котором будут считаны данные и вычислены предсказания.
                Если None, выбирается CUDA, если доступна, иначе CPU.
        threshold: порог для бинаризации вероятностей.

    Returns:
        probs: np.ndarray формы (N, C) — вероятности для каждого класса.
        preds: np.ndarray формы (N, C) — бинарные предсказания (0 или 1).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    all_probs = []
    all_preds = []

    with torch.no_grad():
        for X in test_loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()
            all_probs.append(probs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)

    return probs, preds
