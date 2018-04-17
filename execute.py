import tensorflow as tf
import numpy as np
import pandas as pd
from dataGen import genEpisode, df_trading_day, upperBound, mu_time, std_time
from environment import Policy, Environment


alpha = 1e-3
optimizer_p = tf.train.AdamOptimizer(alpha)
env = Environment()
obsSize = env.obsSize
actSize = env.actSize

sess = tf.Session()
actor = Policy(obsSize, actSize, sess, optimizer_p)

saver = tf.train.Saver()
saver.restore(sess, './model/parameters.ckpt')
print('Model Restored')

adv = []

iterations = 100

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
        
        
    adv.append(env.averagePrice - env.vwap)
    ACTS = np.array(ACTS)
    VALS = np.array(VALS)
    AP = np.sum(ACTS*VALS)/20
    adv.append(AP - vwap)

print(adv)
    