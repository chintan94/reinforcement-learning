# Build Volume profile

import requests
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn import linear_model as linear_model
from pathlib2 import Path
import os

url = 'https://www.alphavantage.co/query?'
query = {'apikey': 'VJRBTK93ZW57CPO6', 'function': 'TIME_SERIES_INTRADAY', 'symbol': 'AAPL', 'interval': '1min', \
         'outputsize': 'full'}
for key, value in query.items():
    url += '&' + key + '=' + value

print("Loading the data (courtesy of AlphaVantage) from " + str(url))

# If the file exists, load it from our cache instead of scraping the web for the data.
file_name = 'minuteData.txt'
print(file_name)
if os.path.isfile(file_name):
    print("Pulling data from file.")
    fp = open(file_name, 'r')
    data = json.load(fp, object_pairs_hook=OrderedDict, encoding='utf-8')
    fp.close()
else:
    print("Pulling data off the web.")
    # Grab each page (it's a .txt file, so we'll treat it as one big string).
    page = requests.get(url)
    #data = page.read()
    data = page.json(encoding='utf-8', object_pairs_hook=OrderedDict)
    # Save the file to the cwd.
    fp = open(file_name, 'w')
    json.dump(data, fp, indent=4, separators=(',', ':'))
    fp.close()

frame_data = {'open': [], 'close': [], 'high': [], 'low': [], 'volume': []}
timestamp = [] 

for key, value in data['Time Series (1min)'].items():
    timestamp.append(pd.to_datetime(key, format='%Y-%m-%d %H:%M:%S'))
    frame_data['open'].append(float(data['Time Series (1min)'][key]['1. open']))
    frame_data['close'].append(float(data['Time Series (1min)'][key]['4. close']))
    frame_data['high'].append(float(data['Time Series (1min)'][key]['2. high']))
    frame_data['low'].append(float(data['Time Series (1min)'][key]['3. low']))
    frame_data['volume'].append(float(data['Time Series (1min)'][key]['5. volume']))

raw_data = pd.DataFrame(data=frame_data, index=timestamp)

# flip the series
raw_data = raw_data.sort_index(ascending = True)
# add a minute bin
# US start of date, calc in minutes
start_of_day = (9 * 60) + 30
# get the time for each bin in minutes and subtract 9:30
raw_data[ 'minute_bars' ] = (raw_data.index.hour * 60) + raw_data.index.minute - start_of_day

# now trim anything beyond bin 384
raw_data = raw_data[ raw_data.minute_bars <= 384 ]
totl_volume = raw_data.groupby( [ raw_data.index.date ] ).tail( 1 )[ 'volume' ]
raw_data[ 'accum_volume' ] = raw_data.groupby( [ raw_data.index.date ] ).cumsum()[ 'volume' ] 
raw_data[ 'accum_pct' ] = raw_data.groupby( [ raw_data.index.date ] )[ 'accum_volume' ].transform( lambda x: x / x.iloc[ -1 ] )

# training data for building the VWAP profile
bars = raw_data[ '03-09-2018':'03-19-2018' ].copy()

# arrange our data
minute_bars = bars[ 'minute_bars' ]
X = pd.DataFrame( { 'bin': minute_bars, 
                    'bin2' : minute_bars**2, 
                    'bin3' : minute_bars**3, 
                    'bin4' : minute_bars**4, 
                    'bin5' : minute_bars**5 } )
y = bars[ 'accum_pct' ]

# do the regression
lm = linear_model.LinearRegression()
model = lm.fit( X, y )
predictions = lm.predict( X )
# now do the regression with no intercept
lm2 = linear_model.LinearRegression( fit_intercept = False )
model = lm2.fit( X, y )
predictions = lm2.predict( X )

def vwap_target( bar_num, coefs ):
    return ( coefs[ 0 ] * bar_num + 
             coefs[ 1 ] * bar_num**2 + 
             coefs[ 2 ] * bar_num**3 +
             coefs[ 3 ] * bar_num**4 +
             coefs[ 4 ] * bar_num**5 )

bins = np.arange(0,385)
target_pct_regr = vwap_target( bins, lm2.coef_ )
target_pct_regr = target_pct_regr/target_pct_regr[-1]

file_name = 'targetPercentage'
outfile = open(file_name, 'wb')

np.save(outfile, target_pct_regr)

outfile.close()