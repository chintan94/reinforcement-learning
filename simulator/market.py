# This program provides random market participants to the OrderBook Simulator
# Assumptions are --
# The number of market participants in the buy and sell side evolve according to Brownian Motions
# The two motions are conditionally independent 
# with independence condition being that market momentum is known.
# The market momentum is encoded in the starting state

# Every call to the program takes input as previous state
# And outputs buy and sell orders of different participants

# Brownian motion form --
# X(0) = X_0
# X(t + dt) = X(t) + N(0, (delta)^2*dt; t, t+dt)

import numpy as np
from scipy.stats import norm

# Function, Given previous book state (Number of buy orders, Best Bid, Number of sell orders, Best Ask)
# Output new orders fired for next time instant
# One more hyper-parameter as Input - market_prob
# This hyper-parameter controls the order book evolution profile

def simulateOrders(N_b, BB, N_s, BA, market_prob = 0.5):
    
    # process parameters
    delta = (N_b + N_s)/300.0
    dt = 0.1

    # initial conditions
    X_b = N_b/2
    X_s = N_s/2

    num_buy = X_b + int(norm.rvs(scale = delta**2*dt))
    num_sell = X_s + int(norm.rvs(scale = delta**2*dt))

    price_resolution = 0.05  # smallest unit of price
    maxMarketSize = 100
    maxLimitSize = 500
    limitRange = 0.1   # limit orders between the range of 10% of current value 
    # Orders for each participant
    # with a probability of market_prob, orders are market orders
    num_market_buy = int(market_prob*num_buy)
    num_limit_buy = num_buy - num_market_buy

    market_buy_sizes = [int(i) for i in maxMarketSize*np.random.random_sample(num_market_buy)]
    limit_buy_sizes = [int(i) for i in maxLimitSize*np.random.random_sample(num_limit_buy)]
    limit_buy_prices = [round(i, 2) for i in price_resolution*np.random.random_integers(BA*(1 - limitRange)/price_resolution, BA/price_resolution, num_limit_buy)]
    limit_buy_orders = [(limit_buy_sizes[i], limit_buy_prices[i]) for i in range(num_limit_buy)]

    num_market_sell = int(market_prob*num_sell)
    num_limit_sell = num_sell - num_market_sell

    market_sell_sizes = [int(i) for i in maxMarketSize*np.random.random_sample(num_market_sell)]
    limit_sell_sizes = [int(i) for i in maxLimitSize*np.random.random_sample(num_limit_sell)]
    limit_sell_prices = [round(i, 2) for i in price_resolution*np.random.random_integers(BB/price_resolution, BB*(1  + limitRange)/price_resolution, num_limit_sell)]
    limit_sell_orders = [(limit_sell_sizes[i], limit_sell_prices[i]) for i in range(num_limit_sell)]

    return(market_buy_sizes, limit_buy_orders, market_sell_sizes, limit_sell_orders)
