from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
import wandb

BATCH_SIZE = 64
DATA_DIR = Path('datasets')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT = "SentimentAnalysis"

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
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

def evaluate_model(model_name, test_data_path):
    wandb.init(project=PROJECT, name=f"{model_name}_eval")

    # Load dataset
    ds = load_dataset(test_data_path, DEVICE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE)

    input_dim = ds.tensors[0].shape[1]
    model = Classifier(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(torch.load(f"{model_name}_final.pth", map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)

    print(f"📊 Results for {model_name}")
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, digits=3))
    print("Confusion Matrix:")
    print(cm)

    wandb.log({
        'test_accuracy': acc,
        'test_precision': precision,
        'test_recall': recall,
        'test_f1': f1,
        'confusion_matrix': wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_labels,
            preds=all_preds,
            title="Confusion Matrix"
        )
    })

    wandb.finish()

def main():
    model_names = [
        "dtm_model_v0",
        "tfidf_model_v0",
        "curated_model_v0"
    ]

    for name in model_names:
        test_path = DATA_DIR / f"test_{name.split('_')[0]}.parquet"
        if not test_path.exists():
            print(f"⚠️ Skipping {name}, test file not found: {test_path}")
            continue
        evaluate_model(name, test_path)

if __name__ == "__main__":
    main()

