import tensorflow as tf
import numpy as np
import pandas as pd
from environment import Policy, Environment
import time
import sys



alpha = 1e-3
numtrajs = 10
iterations = 100

env = Environment()
obsSize = env.obsSize
actSize = env.actSize

sess = tf.Session()

# optimizer
optimizer_p = tf.train.AdamOptimizer(alpha)

# initialize networks
# if command line parameter is given as '/gpu:0', construct the graph for gpu, else construct for cpu
if (len(sys.argv) > 1 and sys.argv[1] == '/gpu:0'):
    actor = Policy(obsSize, actSize, sess, optimizer_p, sys.argv[1])
else:
    actor = Policy(obsSize, actSize, sess, optimizer_p)


# initialize tensorflow graphs
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, './model/parameters.ckpt')
print('Model Restored')


adv = []
start_time = time.time()
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

        obs, done = env.reset()

        while not done:
            prob = actor.compute_prob(np.expand_dims(obs, 0))
            action = np.random.choice(actSize, p=prob.flatten())
            #action = np.argmax(prob)
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

""" end_time = time.time()
train_time = end_time - start_time

fp = open('train_time.txt', 'w')
fp.write(str(train_time))
fp.close()
 """

save_path = saver.save(sess, './model/parameters.ckpt')
print('Model saved in path ' + str(save_path))