
class TradingBot:

    def __init__(self, starting_balance: int, start_date, end_date):

        self.stocks = ['EWI', 'EWJ', 'FCX', 'GRPN', 'KR', 'NVDA', 'TWX']

        self.balance = starting_balance

        self.date = start_date
        self.end_date = end_date


    def buy(self, date, symbol, price, score, headline):
        target_shares = max(1, )






def main():

    starting_balance = 100_000

    bot = TradingBot(starting_balance)




if __name__ == '__main__':
    main()

