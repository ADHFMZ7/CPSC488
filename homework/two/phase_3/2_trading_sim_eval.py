import pandas as pd
from pathlib import Path
import torch
from torch import nn

DATA_DIR = Path('datasets')

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

class TradingBot:

    def __init__(self, starting_balance: int, df):

        self.stocks = ['EWI', 'EWJ', 'FCX', 'GRPN', 'KR', 'NVDA', 'TWX']

        self.balance = starting_balance

        self.df = df.sort_values('date')

        self.date = self.df.date.iloc[0]
        print(self.date)
        self.end_date = self.df.date.iloc[-1]
        print(self.end_date)

        # device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # ---- Load model ----
        model_path = Path('dtm_model_v1_final.pth')  # adjust name as needed
        sample_vec = np.stack(df.news_vector.values)[0]
        input_dim = sample_vec.shape[0]
        hidden_dim = 128
        num_classes = 7

        self.model = Classifier(input_dim, hidden_dim, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def run(self):


        for date, article in self.df.groupby('date'):
            
            x = article.new_vector # < This is a numpy array I think?

            pred_impact = self.model(x)

            

        # Get all vectors on that date

    def log(self):

        pass

def main():

    starting_balance = 100_000

    df = pd.read_parquet(DATA_DIR/'bot_dataset.parquet')
    bot = TradingBot(starting_balance, df)

    bot.run()

if __name__ == '__main__':
    main()

