import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import wandb

from model import MLP

BATCH_SIZE = 128
LEARNING_RATE=1e-3
WEIGHT_DECAY = 1e-4

def create_dataloader(df: pd.DataFrame, device: torch.device, batch_size: int = 128):

    df = df.sort_values(by='date')

    X = np.stack(df.news_vector.values)
    y = (df.impact_score.astype(int) + 3).values

    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    ds = TensorDataset(X_tensor, y_tensor)
    return DataLoader(ds, batch_size), ds.tensors[0].shape[1]


def train_model(df, model_name, project='WordEmbeddings'):
    wandb.init(project=project, name=model_name)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader, input_dim = create_dataloader(df, device, BATCH_SIZE)

    hidden_dim = 128
    num_classes = 7

    model = MLP(input_dim, hidden_dim, num_classes).to(device)

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
        'epochs': 50,
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

