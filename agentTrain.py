import tensorflow as tf
import numpy as np
import pandas as pd
from dataGen import genEpisode, df_trading_day, upperBound, mu_time, std_time
from environment import Policy, Environment



alpha = 1e-3
numtrajs = 10
iterations = 1000

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