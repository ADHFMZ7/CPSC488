from os import path
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
import pandas as pd
import wandb

BATCH_SIZE=64
LEARNING_RATE=1e-3
WEIGHT_DECAY = 1e-4

DATA_DIR = Path('datasets')

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=7):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def normalize(x):
    return x

def load_dataset(path, device):

    df = pd.read_parquet(path)
    df = df.sort_values(by='date')

    X = np.stack(df.news_vector.values)
    y = (df.impact_score.astype(int) + 3).values


    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    Y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    X_tensor = normalize(X_tensor)

    return TensorDataset(X_tensor, Y_tensor)

def train_model(data_path, model_name, project="SentimentAnalysis"):
    wandb.init(project=project, name=model_name) 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = load_dataset(data_path, device)

    loader = DataLoader(ds, batch_size=BATCH_SIZE)
   
    input_dim = ds.tensors[0].shape[1]
    hidden_dim = 128
    num_classes = 7

    model = Classifier(input_dim, hidden_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    wandb.config.update({
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'learning_rate': LEARNING_RATE,
        'weight_decay': WEIGHT_DECAY,
        'batch_size': BATCH_SIZE,
        'optimizer': 'Adam',
        'loss': 'CrossEntropyLoss',
        'epochs': 300,
    })

    for epoch in range(wandb.config.epochs):

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)

        train_loss = total_loss / len(loader)
        train_acc = correct / total

        wandb.log({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc
        })

        print(f"Epoch [{epoch + 1}/{wandb.config.epochs}] "
          f"| Loss: {train_loss:.4f} "
          f"| Accuracy: {train_acc * 100:.2f}%")

    
    torch.save(model.state_dict(), f'{model_name}_final.pth')
    wandb.save(f'{model_name}_final.pth')
    wandb.finish()


def main():
   
    train_model(DATA_DIR/"train_dtm.parquet", "dtm_model_v0")
    train_model(DATA_DIR/"train_tfidf.parquet", "tfidf_model_v0")
    train_model(DATA_DIR/"train_curated.parquet", "curated_model_v0")

if __name__ == '__main__':
    main()
