import pandas as pd

class DataLoader():
    """
    Class loads csv files with financial data
    and executes basic transformation the data into pandas dataframe
    """
    def __init__(self):
        pass

    def load_eurusd_15min(self):
        """
        forex data eurusd --- one pip == 0.0001 usd
        """
        data = pd.read_csv('data/15min1h4h/EURUSD15.csv', sep='\t')
        #data = data[(int(data.shape[0]/2)):]
        data.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        return data, 10000
