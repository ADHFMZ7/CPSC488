import pandas as pd
from pathlib import Path
import torch
from torch import nn
import numpy as np
from math import floor

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

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model_path = Path('dtm_model_v1_final.pth')
        sample_vec = np.stack(df.news_vector.values)[0]
        input_dim = sample_vec.shape[0]
        hidden_dim = 128
        num_classes = 7

        self.model = Classifier(input_dim, hidden_dim, num_classes).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.holdings = {sym: 0 for sym in self.stocks}
        self.trade_log = []

    def run(self):


        for date, group in self.df.groupby('date'):

            x_batch = np.stack(group.news_vector.values)
            x_batch = torch.tensor(x_batch, dtype=torch.float32).to(self.device) 

            with torch.no_grad():
                logits = self.model(x_batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()

            scores = preds - 3

            for (_, row), score in zip(group.iterrows(), scores):
                symbol = row.symbol
                price = row.close
                # headline = getattr(row, "headline", "")
                headline = ""

                self._trade(date, symbol, price, score, headline) 

        self._liquidate()
        self._summarize()

    def _trade(self, date, symbol, price, score, headline=""):
        if score > 0:
            x = max(1, floor((2 * (score / 100) * self.balance) / price))
            cost = x * price
            if cost > self.balance:
                affordable = int(self.balance // price)
                if affordable <= 0:
                    return
                x = affordable
                cost = x * price
            self.balance -= cost
            self.holdings[symbol] += x
            self._log(date, symbol, "Buy", x, price, -cost, self.balance, score, headline)

        elif score < 0:
            shares = self.holdings.get(symbol, 0)
            if shares == 0:
                return
            sell_shares = min(shares, floor(abs(score) / 3 * shares))
            if sell_shares <= 0:
                return
            proceeds = sell_shares * price
            self.balance += proceeds
            self.holdings[symbol] -= sell_shares
            self._log(date, symbol, "Sell", sell_shares, price, proceeds, self.balance, score, headline)


    def _log(self, date, symbol, ttype, shares, price, txn_amount, cash_after, score, headline):
        self.trade_log.append({
            "date": date,
            "symbol": symbol,
            "trade_type": ttype,
            "shares": shares,
            "price": price,
            "txn_amount": txn_amount,
            "cash_after": cash_after,
            "impact_score": score,
            "headline": headline
        })

    def _liquidate(self):
        last_day = self.df.date.max()
        for symbol, shares in self.holdings.items():
            if shares > 0:
                last_price = self.df[self.df.symbol == symbol].iloc[-1].close
                proceeds = shares * last_price
                self.balance += proceeds
                self._log(last_day, symbol, "Final Sell", shares, last_price, proceeds, self.balance, 0, "Final liquidation")
                self.holdings[symbol] = 0

    def _summarize(self):
        trade_df = pd.DataFrame(self.trade_log)
        trade_df.to_csv("trade_log.csv", index=False)

        total_gain = self.balance - 100_000
        total_return = (self.balance / 100_000 - 1) * 100
        summary = pd.DataFrame([{
            "final_balance": self.balance,
            "total_gain_loss": total_gain,
            "total_return_%": total_return
        }])
        summary.to_csv("final_summary.csv", index=False)
        print(summary)


def main():

    starting_balance = 100_000

    df = pd.read_parquet(DATA_DIR/'bot_dataset.parquet')
    bot = TradingBot(starting_balance, df)

    bot.run()

if __name__ == '__main__':
    main()


