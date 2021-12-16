
import multiprocessing as mp
import ctypes
from copy import deepcopy
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow_probability as tfp


import random
import pickle
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from buffer import Record
from actor_critic import _actor_network,_critic_network
import math
from collections import deque

import gym
#import gym_vrep
import pybulletgym
import time

from ou_noise import OUActionNoise

def normalize(val, min, max):
    return (val - min)/(max - min)



def agent(
            env,
            ANN_weights,
            action_dim,
            state_dim,
            gamma,
            T,
            clip,
            epsilon):


    import tensorflow as tf
    import logging
    tf.get_logger().setLevel(logging.ERROR)


    record_temp = []

    At = action = np.zeros(action_dim)

    action_noise = OUActionNoise(action_dim, mu = 0.4)

    ANN = _actor_network(state_dim, action_dim).model()
    ANN.set_weights(ANN_weights)

    state_cache = []
    stack = []

    def forward(tstate):
        action = ANN(tstate)
        eps = max(epsilon, 0.1)

        if random.uniform(0.0, 1.0)>eps:
            action = action[0]
        else:
            action = action[0] + tf.random.normal([action_dim], 0.0, 2*eps)
        return np.clip(action, -1.0, 1.0)



    def process_state(state):
        global TSt, At, Rt
        state_cache.append(state)
        At = action = np.zeros(action_dim)
        if len(state_cache)>1:
            #delta = 2*(state - state_cache[-2])
            #TSt = tf.concat([state, delta], axis=1)
            TSt = state
            At = action = forward(TSt)
        return np.array(action)


    def process_reward(reward, state_next):
        global TSt, At, Rt
        if len(state_cache)>1:
            Rt = reward
            #delta = 2*(state_next - state_cache[-1])
            #TSt_ = tf.concat([state_next, delta], axis=1)
            TSt_ = state_next
            stack.append([TSt, At, Rt, TSt_])


    def update_buffer():
        t_last = len(stack)
        t_last_clipped = t_last - clip
        if t_last>clip:
            for t, (TSt,At,Rt,TSt_) in enumerate(stack):
                if t<t_last_clipped:
                    Qt = 0.0
                    discount = 1.0
                    for k in range(t, t+clip):
                        Qt = Qt + stack[k][2]*discount
                        discount *= gamma
                    Qt_ = (Qt - Rt)/gamma + stack[t+clip][2]*discount
                    record_temp.append([TSt,At,Rt,Qt,TSt_,Qt_])


    done = False
    done_cnt = 0
    stack = []

    state = np.array(env.reset(), dtype='float32').reshape(1, state_dim)

    for t in range(1, T):
        if t%200 == 0: ANN.set_weights(ANN_weights)

        action = process_state(state)
        state_next, reward, done, info = env.step(action)
        state_next = np.array(state_next).reshape(1, state_dim)

        if done:
            if reward<=-100.0 or reward>=100.0:
                reward = reward/clip

            if done_cnt>=clip:
                done_cnt = 0
                break
            else:
                done_cnt += 1


        state = state_next
        process_reward(reward, state_next)

        if t >= T-1:
            break


    update_buffer()

    return record_temp



class DDPG():
    def __init__(self,
                 env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 max_buffer_size =10000, # maximum transitions to be stored in buffer
                 batch_size =64, # batch size for training actor and critic networks
                 max_time_steps = 1000 ,# no of time steps per epoch
                 clip = 25,
                 discount_factor  = 0.99,
                 explore_time = 2000, # time steps for random actions for exploration
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 n_episodes = 1000):# no of episodes to run


        #############################################
        # --------------- Parametres-----------------#
        #############################################
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.gamma = discount_factor  ## discount factor
        self.explore_time = explore_time
        self.act_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.n_episodes = n_episodes

        self.env = env
        self.observ_min = self.env.observation_space.low
        self.observ_max = self.env.observation_space.high
        self.action_dim = action_dim = env.action_space.shape[0]

        self.action_noise = OUActionNoise(action_dim, mu = 0.4)


        self.state_cache = []
        self.reward_cache = []
        self.TSt = None
        self.At = None
        self.Rt = None
        self.stack = []
        self.s_x = 0.0
        self.epsilon = 1.0
        self.Q = [-9999.0]

        self.N_stm = 10
        observation_dim = len(env.reset())
        self.state_dim = state_dim = observation_dim

        self.exp_weights = np.ones((self.N_stm,1,observation_dim), dtype='float32')*np.exp(-(2/self.N_stm)*np.arange(0, self.N_stm, 1, dtype='float32')).reshape((self.N_stm, 1, 1))

        self.Q_log = []
        self.dq_da_history = []
        self.clip = clip
        self.T = max_time_steps + self.clip  ## Time limit for a episode
        self.stack_steps = max_time_steps


        self.ANN_Adam = Adam(self.act_learning_rate)
        self.QNN_Adam = Adam(self.critic_learning_rate)

        self.record = Record(self.max_buffer_size, self.batch_size)

        self.ANN = _actor_network(self.state_dim, self.action_dim).model()

        self.QNN_pred = _critic_network(self.state_dim, self.action_dim).model()
        self.QNN_pred.compile(loss='mse', optimizer=self.QNN_Adam)

        self.QNN_target = _critic_network(self.state_dim, self.action_dim).model()
        self.QNN_target.set_weights(self.QNN_pred.get_weights())
        self.QNN_target.compile(loss='mse', optimizer=self.QNN_Adam)

        self.agents_running = False

        self.training = False

        #############################################
        #----Action based on exploration policy-----#
        #############################################

    def forward(self, tstate):
        action = self.ANN(tstate)
        epsilon = max(self.epsilon, 0.1)
        if random.uniform(0.0, 1.0)>self.epsilon:
            action = action[0]
        else:
            action = action[0] + tf.random.normal([self.action_dim], 0.0, 2*epsilon)
        return np.clip(action, -1.0, 1.0)

    def process_state(self, state):
        self.state_cache.append(state)
        action = np.zeros(self.action_dim)
        if len(self.state_cache)>1:
            #delta = 2*(state - self.state_cache[-2])
            #self.TSt = tf.concat([state, delta], axis=1)
            self.TSt = state
            self.At = action = self.forward(self.TSt)
        return np.array(action)


    def process_reward(self, reward, state_next):
        if len(self.state_cache)>1:
            self.Rt = reward
            #delta = 2*(state_next - self.state_cache[-1])
            #self.TSt_ = tf.concat([state_next, delta], axis=1)

            self.TSt_ = state_next

            self.stack.append([self.TSt, self.At, self.Rt, self.TSt_])


    def update_buffer(self):
        t_last = len(self.stack)
        t_last_clipped = t_last - self.clip
        if t_last>self.clip:
            for t, (TSt,At,Rt,TSt_) in enumerate(self.stack):
                if t<t_last_clipped:
                    Qt = 0.0
                    discount = 1.0
                    for k in range(t, t+self.clip):
                        Rk = self.stack[k][2]
                        Qt = Qt + discount*Rk
                        discount *= self.gamma
                    Qt_ = (Qt - Rt)/self.gamma + self.stack[t+self.clip][2]*discount
                    self.record.add_experience(TSt,At,Rt,Qt,TSt_,Qt_)
            self.stack = self.stack[-self.clip:]


    #############################################
    # --------------Update Networks--------------#
    #############################################

    def ddpg_backprop(self, actor, critic, optimizer, tstates_batch, dq_da_history, N):
        #t = tf.convert_to_tensor(tstates_batch, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            a = actor(tstates_batch)
            tape.watch(a)
            q = critic([tstates_batch, a])
        dq_da = tape.gradient(q, a)
        policy_gradient = dq_da

        dq_da_history.append(policy_gradient)

        if len(dq_da_history)>N:
            dq_da_history = dq_da_history[-N-1:]
            sma_prev = np.mean(dq_da_history[-N-1:-1], axis=0)
            sma_current = np.mean(dq_da_history[-N:], axis=0)

            advantage = policy_gradient - sma_current

            policy_gradient = sma_prev + advantage
            policy_gradient = np.abs(policy_gradient)*np.tanh(policy_gradient)

        with tf.GradientTape(persistent=True) as tape:
            a = actor(tstates_batch)
            theta = actor.trainable_variables
        da_dtheta = tape.gradient(a, theta, output_gradients=-policy_gradient)
        optimizer.apply_gradients(zip(da_dtheta, actor.trainable_variables))


    def bootstrap(self):
        TSt, At, Rt, Q, TSt_, Qt_ = self.record.sample_batch()
        At_ = self.ANN(TSt_)
        Q_ = self.QNN_target([TSt_, At_])
        tempR = np.abs(0.1*Rt/math.log(0.1))
        tempQ_ = np.abs(0.1*Q_/math.log(0.1))
        Q = (Rt - tempR*np.log(np.abs(At))) + self.gamma*(Q_- tempQ_*np.log(np.abs(At_)))

        #self.QNN_pred.train_on_batch([TSt, At], Q)
        self.train_on_batch(self.QNN_pred, TSt, At, Q)
        self.ddpg_backprop(self.ANN, self.QNN_pred, self.ANN_Adam, TSt, self.dq_da_history, 10)

    def train_on_batch(self,QNN,TSt,At,Q):
        with tf.GradientTape() as tape:
            mse = (1/2)*(Q-QNN([TSt, At]))**2
        gradient = tape.gradient(mse, QNN.trainable_variables)
        self.QNN_Adam.apply_gradients(zip(gradient, QNN.trainable_variables))


    def networks_update(self):
        TSt, At, Rt, Q, TSt_, Qt_ = self.record.sample_batch()
        self.Q_network_target_update(TSt,At,Q)
        self.Q_network_pred_update(TSt,At,Rt,TSt_,Qt_)



    def Q_network_target_update(self,TSt,At,Q):
        #self.QNN_target.train_on_batch([TSt, At], Q)
        self.train_on_batch(self.QNN_target, TSt, At, Q)
        self.ddpg_backprop(self.ANN, self.QNN_target, self.ANN_Adam, TSt, self.dq_da_history, 10)

    def Q_network_pred_update(self,TSt,At,Rt,TSt_,Qt_):
        At_ = self.ANN(TSt_)
        Q_ = self.QNN_target([TSt_, At_])
        Q = Rt + self.gamma*(Q_+Qt_)/2

        #self.QNN_pred.train_on_batch([TSt, At], Q)
        self.train_on_batch(self.QNN_pred, TSt, At, Q)
        self.ddpg_backprop(self.ANN, self.QNN_pred, self.ANN_Adam, TSt, self.dq_da_history, 10)


    def sync_target(self):
        self.QNN_target.set_weights(self.QNN_pred.get_weights())

    def clear_stack(self):

        self.state_cache = []
        self.reward_cache = []
        self.stack = []


    def save(self):
        result = 0
        while result<10:
            time.sleep(0.01)
            try:
                result += 1
                self.ANN.save('./models/actor.h5')
                self.QNN_pred.save('./models/critic_pred.h5')
                self.QNN_target.save('./models/critic_target.h5')
                #with open('buffer', 'wb') as file:
                    #pickle.dump({'buffer': self.record}, file)
                return
            except:
                pass



    def epsilon_dt(self):
        self.s_x += 0.01
        self.epsilon = math.exp(-1.0*self.s_x)*math.cos(self.s_x)


    def train(self):

        ANN_weights = self.ANN.get_weights()
        shared_weights = []
        for weights in ANN_weights:
            _shared_array = mp.Array(ctypes.c_double, weights.flatten(), lock=False)
            shared_array = np.frombuffer(_shared_array, dtype='double').reshape(weights.shape)
            shared_weights.append(shared_array)

        self.agents_cnt = 0
        self.pool = mp.Pool(4)

        def pool_restart():
            try:
                if self.agents_running:
                    [agent.wait() for agent in self.agents]
                    self.pool.close()
            except:
                pass
            self.pool = mp.Pool(4)
            self.agents = [self.pool.apply_async(agent, args=(self.env, shared_weights, self.action_dim, state_dim, self.gamma, self.T, self.clip, self.epsilon),
                                callback=agents_record) for _ in range(4)]
            self.agents_running = True


        def agents_record(record):
            for data in record:
                TSt,At,Rt,Qt,TSt_,Qt_ = data[0], data[1], data[2], data[3], data[4], data[5]
                self.record.add_experience(TSt,At,Rt,Qt,TSt_,Qt_)
                if self.agents_cnt>=3:
                    self.agents_running = False
                    self.agents_cnt = 0
                elif self.agents_running:
                    self.agents_cnt += 1


        with open('Scores.txt', 'w+') as f:
            f.write('')


        state_dim = len(self.env.reset())

        cnt = 1


        score_history = []
        for episode in range(self.n_episodes):

            self.episode = episode

            done = False
            score = 0.0
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)
            self.epsilon_dt()

            pool_restart()
            t = 0
            done_cnt = 0

            DONE = False
            while not DONE:
            #for t in range(self.T):

                if t>=(self.T//2) and t%(self.T//2)==0:
                    if not self.agents_running: pool_restart()

                self.env.render()

                action = self.process_state(state)
                state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                state_next = np.array(state_next).reshape(1, state_dim)


                if done:
                    if reward <=-100 or reward >=100:
                        reward = reward/self.clip

                    if done_cnt>self.clip:
                        DONE = True
                        done_cnt = 0
                        break
                    else:
                        done_cnt += 1

                score += reward
                state = state_next
                self.process_reward(reward, state_next)

                if len(self.stack)>=(10 + self.clip) and cnt%10 == 0:
                    self.update_buffer()


                if len(self.record.buffer)>self.batch_size:
                    if cnt%(1+self.explore_time//cnt)==0:
                        if t%2==0:
                            self.networks_update()
                        else:
                            self.sync_target()
                            self.bootstrap()
                            shared_weights = deepcopy(self.ANN.get_weights())


                t += 1
                cnt += 1

            self.update_buffer()


            self.clear_stack()


            if episode>=20 and episode%20==0:
                self.save()


            score_history.append(score)
            avg_score = np.mean(score_history[-10:])
            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')

            print('%d: %f, %f, | %f | record size %d' % (episode, score, avg_score, self.epsilon, len(self.record.buffer)))



    def test(self):

        with open('Scores.txt', 'w+') as f:
            f.write('')

        self.epsilon = 0.0
        state_dim = len(self.env.reset())
        cnt = 1
        score_history = []

        for episode in range(self.n_episodes):

            self.episode = episode

            done = False
            score = 0.0
            state = np.array(self.env.reset(), dtype='float32').reshape(1, state_dim)

            t = 0
            done_cnt = 0

            for t in range(self.T):

                self.env.render()

                action = self.forward(state)
                state_next, reward, done, info = self.env.step(action)  # step returns obs+1, reward, done
                state_next = np.array(state_next).reshape(1, state_dim)


                if done:
                    if reward <=-100 or reward >=100:
                        reward = reward/self.clip

                    if done_cnt>self.clip:
                        done_cnt = 0
                        break
                    else:
                        done_cnt += 1

                score += reward
                state = state_next

                cnt += 1


            score_history.append(score)
            avg_score = np.mean(score_history[-10:])

            with open('Scores.txt', 'a+') as f:
                f.write(str(score) + '\n')

            print('%d: %f, %f ' % (episode, score, avg_score))


#env = gym.make('Pendulum-v1').env
#env = gym.make('LunarLanderContinuous-v2').env
#env = gym.make('HumanoidMuJoCoEnv-v0').env
#env = gym.make('BipedalWalkerHardcore-v3').env
env = gym.make('BipedalWalker-v3').env
#env = gym.make('HalfCheetahMuJoCoEnv-v0').env


ddpg = DDPG(     env , # Gym environment with continous action space
                 actor=None,
                 critic=None,
                 buffer=None,
                 max_buffer_size =2000000, # maximum transitions to be stored in buffer
                 batch_size = 100, # batch size for training actor and critic networks
                 max_time_steps = 2000,# no of time steps per epoch
                 clip = 700,
                 discount_factor  = 0.99,
                 explore_time = 2000,
                 actor_learning_rate = 0.0001,
                 critic_learning_rate = 0.001,
                 n_episodes = 1000000) # no of episodes to run


ddpg.train()
