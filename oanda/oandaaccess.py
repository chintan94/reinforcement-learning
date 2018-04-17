import configparser
import oandapyV20 as opy
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.instruments as instruments
import pandas as pd

config = configparser.ConfigParser()
config.read('oanda.cfg')

api = opy.API(access_token=config['oanda']['access_token'])

# r = trades.TradesList(config['oanda']['account_id'])   # should be empty


params = {'count': 5, 'granularity': 'S5', 'from': '2018-04-13T14:30:15.000000000Z'}
# params = {}
response = instruments.InstrumentsCandles(instrument = 'EUR_USD', params=params)
# r = instruments.InstrumentsOrderBook(instrument = 'EUR_USD', params = params)

data = api.request(response)


df = pd.DataFrame(data['candles']).set_index('time')

print(df)


