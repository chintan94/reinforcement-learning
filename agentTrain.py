# ---------------------------------RL agent ---------------------------------------
# State is defined as -- time_remaining(in secs), quantity remaining(in lotSizes), current_min, "Momentum", bid_price
# Action is an Integer <= quantity remaining
# Action taken specifies the number of lot sizes the agent decides to sell at Bid Price
# Orders are assumed to be immediate

# The problem to optimize for the RL agent is --
# Sell 2000 shares in lots at bid price of a given tick
# with minimum lot size 100 and maximum lot size 400
# Therefore, action space is discrete of size 5 (0, 1, 2, 3 , 4)

# "Momentum" for our purposes is the sum of product of last 10 returns and their volumes

# Policy gradient method is used for optimization
# The policy network has input as momentum(un-standardized), bid_price(standardized), quantity_remaining(in lotSizes)/time_remaining(in secs), current bar_time(standardized minutes)

import tensorflow as tf
import numpy as np
import pandas as pd
from dataGen import genEpisode, df_trading_day, upperBound, mu_time, std_time
from environment import Policy, Environment



alpha = 1e-3
numtrajs = 1
iterations = 1

env = Environment()
obsSize = env.obsSize
actSize = env.actSize

sess = tf.Session()

# optimizer
optimizer_p = tf.train.AdamOptimizer(alpha)

# initialize networks
actor = Policy(obsSize, actSize, sess, optimizer_p)

# initialize tensorflow graphs
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

""" # Difference between VWAP for the minute and average price achieved when random actions are taken by the agent(not trained) in 100 iterations
adv = []

for ite in range(iterations):
    
    ACTS = []
    VALS = []
    obs = env.reset()
    done = False

    while not done:
        prob = actor.compute_prob(np.expand_dims(obs, 0))
        action = np.random.choice(actSize, p = prob.flatten())
        newobs, reward, done, vwap = env.step(action)
        obs = newobs
        ACTS.append(env.action)
        VALS.append(env.episodeSlice.iloc[env.index]['Bid_Price'])

    ACTS = np.array(ACTS)
    VALS = np.array(VALS)
    AP = np.sum(ACTS*VALS)/20
    adv.append(AP - vwap)

print(adv) """

adv = []
for ite in range(iterations):
    
    # trajectory records for batch update
    OBS = []
    ACTS = []
    VALS = []
    
    for num in range(numtrajs):
        # record for each trajectory
        obss = []
        acts = []
        rews = []

        obs = env.reset()
        done = False

        while not done:
            prob = actor.compute_prob(np.expand_dims(obs, 0))
            action = np.random.choice(actSize, p=prob.flatten())
            newobs, reward, done, _ = env.step(action)

            obss.append(obs)
            acts.append(action)
            rews.append(reward)
            obs = newobs
        adv.append(env.averagePrice - env.vwap)
        VALS += rews
        OBS += obss
        ACTS += acts

    VALS = np.array(VALS)
    OBS = np.array(OBS)
    ACTS = np.array(ACTS)

    actor.train(OBS, ACTS, VALS)

adv = np.array(adv)
save_path = saver.save(sess, './model/parameters.ckpt')
print('Model saved in path ' + str(save_path))