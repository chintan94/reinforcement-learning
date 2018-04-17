
# coding: utf-8

# # Order Book Simulator v1

# In[1]:

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict, deque
from market import simulateOrders


# In[11]:

# Simple order book:

buy_orders = {} # key is price, value is size
sell_orders = {}
buy_market = deque()
sell_market = deque()
trades = []


def show_book():
    buy_qty = []
    buy_price = []
    sell_price = []
    sell_qty = []
    for key in sorted( buy_orders, reverse = True ):
        buy_price.append( key )
        buy_qty.append(int( buy_orders[key]))
    for key in sorted( sell_orders ):
        sell_price.append( key)
        sell_qty.append(int(sell_orders[key]))
    
    d = OrderedDict( B_QTY = buy_qty, B_PX = buy_price, S_PX = sell_price, S_QTY = sell_qty, B_MKT = buy_market, S_MKT = sell_market )
    display_book = pd.DataFrame( OrderedDict( [ ( k, pd.Series( v ) ) for k, v in d.items() ] ) )
    print( display_book )

      
# return best bid and offer
def get_BBO():
    return ( max( buy_orders.items() ), 
            min( sell_orders.items() ) )

# new market order arrives
def new_market_order(order_side, order_qty):
    # side = s | b, fail on anything else
    if not ( order_side in ["s", "b"] ):
        print( "Side '" + order_side + "' not supported." )
        return
    # price: float
    # qty: integer
    remaining_order_qty = order_qty

    if (order_side == 's'):
        # loop through the buy order_book
        for key in sorted(buy_orders, reverse = True ):
            buy_qty = buy_orders[ key ]
            buy_price = key

            trade_qty = min ( buy_qty, remaining_order_qty )
            trade_price = buy_price

            if (buy_qty >= remaining_order_qty):
                # we have exhausted our order quantity, update the resting quantity
                if ( buy_qty == remaining_order_qty ):
                    # nothing left at this level, remove it
                    del buy_orders[ buy_price ]

                else:
                    # update residual order book quantity
                    buy_orders[ buy_price ] -= remaining_order_qty                    
                trades.append( ( trade_price, trade_qty ) )
                remaining_order_qty = 0
                # we are done
                break
            else:
                del buy_orders[buy_price]
                remaining_order_qty -= buy_qty
                trades.append((trade_price, trade_qty))
        if remaining_order_qty > 0:
            sell_market.append(remaining_order_qty)

    else:
        # loop through the sell orders
        for key in sorted(sell_orders):
            sell_price = key
            sell_qty = sell_orders[ key ]

            trade_qty = min ( sell_qty, remaining_order_qty )
            trade_price = sell_price

            if (sell_qty >= remaining_order_qty):
                # we have exhausted our order quantity, update the resting quantity
                if ( sell_qty == remaining_order_qty ):
                    # nothing left at this level, remove it
                    del sell_orders[ sell_price ]

                else:
                    # update residual order book quantity
                    sell_orders[ sell_price ] -= remaining_order_qty                    
                trades.append( ( trade_price, trade_qty ) )
                remaining_order_qty = 0
                # we are done
                break
            else:
                del sell_orders[sell_price]
                remaining_order_qty -= sell_qty
                trades.append((trade_price, trade_qty))
        if remaining_order_qty > 0:
            buy_market.append(remaining_order_qty)


# add a new limit order to the book
def new_limit_order(order_side, order_price, order_qty):
    # side = s | b, fail on anything else
    if not ( order_side in ["s", "b"] ):
        print( "Side '" + order_side + "' not supported." )
        return
    # price: float
    # qty: integer
    remaining_order_qty = order_qty
    
    # it's a sell order
    if (order_side == 's'):
        # loop through buy_market list if any
        while(len(buy_market) > 0):
            trade_qty = min(remaining_order_qty, buy_market[0])
            trade_price = order_price
            remaining_order_qty -= buy_market.pop()
            trades.append((trade_price, trade_qty))
            if (remaining_order_qty <= 0):
                return
        # loop thru buy orders
        for key in sorted(buy_orders, reverse = True ):
            
            buy_price = key
            buy_qty = int(buy_orders[ key ])
            # check for price at this level
            if order_price <= buy_price: # we have at least one match
                trade_qty = min ( buy_qty, remaining_order_qty )
                trade_price = buy_price
                # update order book
                # remove or reduce order 
                if buy_qty >= remaining_order_qty:
                    # we have exhausted our order quantity, update the resting quantity
                    if ( buy_qty == remaining_order_qty ):
                        # nothing left at this level, remove it
                        del buy_orders[ buy_price ]
                    else:
                        # update residual order book quantity
                        buy_orders[ buy_price ] -= remaining_order_qty
                    trades.append( ( trade_price, trade_qty ) )
                
                    break # we're done
                else: # we have more order left, remove this price level in the book and continue
                    # remove buy order 
                    del buy_orders[key]
                    # decrement sell qty
                    remaining_order_qty -= trade_qty
                    # record trade
                    trades.append( ( trade_price, trade_qty ) )
                    continue
            else: # prices didn’t match, so add it to the sell side
                if order_price in sell_orders:
                    sell_orders[ order_price ] += remaining_order_qty
                else:
                    
                    sell_orders[ order_price ] = remaining_order_qty
                break 

    # it's a buy  
    else:
        # loop through sell_market list if any
        while(len(sell_market) > 0):
            trade_qty = min(remaining_order_qty, sell_market[0])
            trade_price = order_price
            remaining_order_qty -= sell_market.pop()
            trades.append((trade_price, trade_qty))
            if (remaining_order_qty <= 0):
                return
        # loop thru sell orders
        for key in sorted( sell_orders ):
            sell_price = key
            sell_qty = sell_orders[ key ]
            # check for price at this level
            if order_price >= sell_price: # we have at least one match
                trade_qty = min ( sell_qty, remaining_order_qty )
                trade_price = sell_price
                # update order book
                # remove or reduce order 
                if sell_qty >= remaining_order_qty:
                    # we have exhausted our order quantity, update the resting quantity
                    if ( sell_qty == remaining_order_qty ):
                        # nothing left at this level, remove it
                        del sell_orders[ sell_price ]
                    else:
                        # update residual order book quantity
                        sell_orders[ sell_price ] -= remaining_order_qty
                    trades.append( ( trade_price, trade_qty ) )
                    break # we're done
                else: # we have more order left, remove this price level in the book and continue
                    # remove buy order 
                    del sell_orders[key]
                    # decrement sell qty
                    remaining_order_qty -= trade_qty

                    # record trade
                    trades.append( ( trade_price, trade_qty ) )
                    continue
            else: # prices didn’t match, so add it to the sell side
                if order_price in buy_orders:
                    buy_orders[ order_price ] += remaining_order_qty
                else:
                    buy_orders[ order_price ] = remaining_order_qty
                break 
    


def initOrderBook():
    tick_size = 0.05
    init_price = round(tick_size*np.random.randint(500,1000), 2)

    # Populate the order book with 10 random buy and sell orders
    for i in range(10):
        buy_qty = np.random.randint(50, 100)
        sell_qty = np.random.randint(50, 100)
        buy_price = init_price - round(0.05*(i + 1), 2)
        sell_price = init_price + round(0.05*(i + 1), 2)

        buy_orders[buy_price] = buy_qty
        sell_orders[sell_price] = sell_qty

initOrderBook()
show_book()

Time = 1

for i in range(Time):
    market_buy_sizes, limit_buy_orders, market_sell_sizes, limit_sell_orders = simulateOrders(100, 35, 100, 35.5)

    for order_size, order_price in limit_buy_orders:
        new_limit_order('b', order_price, order_size)

    show_book()

    for order_size, order_price in limit_sell_orders:
        new_limit_order('s', order_price, order_size)

    show_book()
    





# dump trades to a DataFrame
# Order Book Metrics
# automate and simulate orders in a book 
# random side at inside market
# random size
# random side and price (scaled to volatility)



