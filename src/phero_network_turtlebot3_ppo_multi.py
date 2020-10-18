#! /usr/bin/env python

# The turtlebot is trained to find the optimal velocity (Twist) with given pheromone data
# The environment is given in the pair script (phero_turtlebot_turtlebot3_ppo.py)
# The expected result is following the pheromone in the most smooth way! even more than ants

#import phero_turtlebot_turtlebot3_ppo
import phero_turtlebot_turtlebot3_repellent_ppo_multi
import numpy as np
import os
import sys
import multiprocessing
# from keras.models import Sequential, Model
# from keras.layers import Dense, Dropout, Input, merge
# from keras.layers.merge import Add, Concatenate
# from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import random
from collections import deque
from utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from distributions import make_pdtype
import os.path as osp
import joblib
import tensorboard_logging
import timeit
import csv
import math
import time
import matplotlib.pyplot as plt
import scipy.io as sio

import logger
#from numba import jit

class AbstractEnvRunner(object):
    '''
    Basic class import env, model and etc (used in Runner class)
    '''
    def __init__(self, env, model, nsteps):
        self.env = env
        self.model = model
        nenv = 1
        self.obs = env.reset()
        self.nsteps = nsteps
        self.states = model.initial_state
        self.dones = [False for _ in range(nenv)]

    def run(self):
        raise NotImplementedError

class PheroTurtlebotPolicy(object):
    '''
    Policy Class
    '''
    #20201009 Detail needs to be modified specifically for phero turtlebot use
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False, deterministic = False): #pylint: disable=W0613
        
        # Assign action as Gaussian Distribution
        self.pdtype = make_pdtype(ac_space)
        self.num_obs = 13
        #print("action_space: {}".format(ac_space))
        with tf.variable_scope("model", reuse=reuse):
            phero_values = tf.placeholder(shape=(None, self.num_obs), dtype=tf.float32, name="phero_values")
            #velocities = tf.placeholder(shape=(None, 2), dtype=tf.float32, name="velocities")

            # Actor neural net
            pi_net = self.net(phero_values)
            # Critic neural net
            vf_h2 = self.net(phero_values)
            vf = fc(vf_h2, 'vf', 1)[:,0]

            self.pd, self.pi = self.pdtype.pdfromlatent(pi_net, init_scale=0.01)

        if deterministic:
            a0 = self.pd.mode()
        else:
            a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None
        self.phero = phero_values
        #self.velocities = velocities

        self.vf = vf

        def step(ob, *_args, **_kwargs):
            '''
            Generate action & value & log probability by inputting one observation into the policy neural net
            '''
            phero = [o for o in ob]

            # lb = [o["laser"] for o in ob]
            # rb = [o["rel_goal"] for o in ob]
            # vb = [o["velocities"] for o in ob]

            a, v, neglogp = sess.run([a0, vf, neglogp0], {self.phero: phero})
            # Action clipping (normalising action within the range (-1, 1) for better training)
            # The network will learn what is happening as the training goes.
            # for i in range(a.shape[1]):
            #     a[0][i] = min(1.0, max(-1.0, a[0][i]))
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            phero = [o for o in ob]
            # lb = [o["laser"] for o in ob]
            # rb = [o["rel_goal"] for o in ob]
            # vb = [o["velocities"] for o in ob]
            return sess.run(vf, {self.phero: phero})

        self.step = step
        self.value = value

    def net(self, phero):
        '''
        Policy Network 
        '''
        # 20201009 Simple neural net. Needs to be modified for better output.
        net = tf.layers.dense(phero, 256, activation=tf.nn.relu)
        net = tf.layers.dense(net, 256, activation=tf.nn.relu)
        net = tf.layers.dense(net, 128, activation=tf.nn.relu)
        #net = tf.layers.dense(net, 1, activation=tf.nn.relu)

        return net

# Class of Actor Critic Model for PPO
class Model(object):
    """
    We use this object to :
    __init__:
    - Creates the step_model
    - Creates the train_model
    - Create placeholders needs for PPO
    train():
    - Make the training part (feedforward and retropropagation of gradients)
    save/load():
    - Save load the model
    """
    def __init__(self, policy, ob_space, ac_space, nbatch_act, nbatch_train,
                 nsteps, ent_coef, vf_coef, max_grad_norm, deterministic=False):
    
        self.sess = sess = self.get_session()

        # Create two models
        ## Actor model for sampling
        print("Pre Model")
        act_model = policy(sess, ob_space, ac_space, nbatch_act, 1, reuse=False, deterministic=deterministic)
        ## Train model for training
        train_model = policy(sess, ob_space, ac_space, nbatch_train, nsteps, reuse=True, deterministic=deterministic)
        print("After Model")
        '''Create Placeholders'''
        A = train_model.pdtype.sample_placeholder([None])
        ADV = tf.placeholder(tf.float32, [None])
        R = tf.placeholder(tf.float32, [None])

        # Keep track of old actor
        OLDNEGLOGPAC = tf.placeholder(tf.float32, [None])
        # Keep track of old critic
        OLDVPRED = tf.placeholder(tf.float32, [None])
        LR = tf.placeholder(tf.float32, [])

        # Clip Range
        CLIPRANGE = tf.placeholder(tf.float32, [])

        neglogpac = train_model.pd.neglogp(A)
        
        # Entropy is used to improve exploration by limiting the premature convergence to suboptimal policy
        entropy = tf.reduce_mean(train_model.pd.entropy())


        ''' LOSS CALCULATION '''
        # Total loss = Policy gradient loss (clipped) - entropy * entropy coefficient + Value coefficient * value loss
        
        # Clip the value to reduce variability during Critic training
        # Get the predicted value
        vpred = train_model.vf
        vpredclipped = OLDVPRED + tf.clip_by_value(train_model.vf - OLDVPRED, - CLIPRANGE, CLIPRANGE)
        # Unclipped loss
        vf_losses1 = tf.square(vpred - R)
        # Clipped loss
        vf_losses2 = tf.square(vpredclipped - R)
        # Average them
        vf_loss = .5 * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))
        
        # Calculate ratio (current policy / old policy)
        ratio = tf.exp(OLDNEGLOGPAC - neglogpac)
        pg_losses = -ADV * ratio
        pg_losses2 = -ADV * tf.clip_by_value(ratio, 1.0 - CLIPRANGE, 1.0 + CLIPRANGE)
        # Final PG loss
        pg_loss = tf.reduce_mean(tf.maximum(pg_losses, pg_losses2))
        approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - OLDNEGLOGPAC))
        clipfrac = tf.reduce_mean(tf.to_float(tf.greater(tf.abs(ratio - 1.0), CLIPRANGE)))
        
        # Calculate total loss
        loss = pg_loss - entropy * ent_coef + vf_loss * vf_coef

        ''' UPDATE THE PARAMETERS USING LOSS '''
        # 1. Get the model parameters
        with tf.variable_scope('model'):
            params = tf.trainable_variables()
        # 2. Calculate the gradients - g from L (L_clip + L_vf + L_s)(theta)
        grads = tf.gradients(loss, params)
        if max_grad_norm is not None:
            grads, _grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        grads = list(zip(grads, params))
        # 3. Build our trainer
        trainer = tf.train.AdamOptimizer(learning_rate=LR, epsilon=1e-5)
        _train = trainer.apply_gradients(grads)

        def reshape2d(arr):
            return np.reshape(arr, (len(arr)*len(arr[0]), len(arr[0][0])), 'F')

        def reshape1d(arr):
            return np.reshape(arr, (len(arr)*len(arr[0])), 'F')

        # Reshaping the state arrays for training (to feed in the neural network)
        # 20201009 needs to rewrite for Phero TUrtlebot use
        def reshape(ids, obs, returns, masks, actions, values, neglogpacs):
            
            
            # ib = np.asarray([[o["id"] for o in iobs] for iobs in obs])
            # lb = np.asarray([[o["laser"] for o in iobs] for iobs in obs])
            # rb = np.asarray([[o["rel_goal"] for o in iobs] for iobs in obs])
            # vb = np.asarray([[o["velocities"] for o in iobs] for iobs in obs])

            #lb = np.reshape(lb, (len(lb)*len(lb[0]), 512, 3), 'F')

            pb = reshape2d(obs)
            #rb = reshape2d(rb)
            #vb = reshape2d(vb)

            ids = reshape1d(ids)

            actions = reshape2d(actions)

            advs = returns - values
            #advs = (advs - advs.mean()) / (advs.std() + 1e-8)

            advs = reshape1d(advs)
            returns = reshape1d(returns)
            neglogpacs = reshape1d(neglogpacs)
            values = reshape1d(values)

            return pb, advs, returns, masks, actions, values, neglogpacs

        
        def train(lr, cliprange, ids, obs, returns, masks, actions, values, neglogpacs, states=None):
            '''
            Put values into placeholders and train the policy neural network
            '''
            pb, advs, returns, masks, actions, values, neglogpacs = reshape(ids, obs, returns, masks, actions, values, neglogpacs)

            td_map = {train_model.phero:pb, A:actions, ADV:advs, R:returns, LR:lr,
                    CLIPRANGE:cliprange, OLDNEGLOGPAC:neglogpacs, OLDVPRED:values}
            
            if states is not None:
                td_map[train_model.S] = states
                td_map[train_model.M] = masks
            #print("here?") 
            return sess.run(
                [pg_loss, vf_loss, entropy, approxkl, clipfrac, _train],
                td_map
            )[:-1]
        

        def save(save_path):
            save_path = self.saver.save(sess, save_path)
            print("Model saved in path: %s" % save_path)

        def restore(restore_path):
            self.saver.restore(sess, restore_path)
            print("Model restored.")
        
        self.loss_names = ['policy_loss', 'value_loss', 'policy_entropy', 'approxkl', 'clipfrac']
        self.train = train
        self.train_model = train_model
        self.act_model = act_model
        self.step = act_model.step
        self.value = act_model.value
        self.initial_state = act_model.initial_state
        self.saver = tf.train.Saver()
        self.save = save
        self.restore = restore
        self.num_obs = act_model.num_obs
        tf.global_variables_initializer().run(session=sess) #pylint: disable=E1101
    
    def get_session(self, config=None):
        """Get default session or create one with a given config"""
        sess = tf.get_default_session()
        if sess is None:
            sess = self.make_session(config=config, make_default=True)
        return sess

    def make_session(self, config=None, num_cpu=None, make_default=False, graph=None):
        """Returns a session that will use <num_cpu> CPU's only"""
        if num_cpu is None:
            num_cpu = int(os.getenv('RCALL_NUM_CPU', multiprocessing.cpu_count()))
        if config is None:
            config = tf.ConfigProto(
                allow_soft_placement=True,
                inter_op_parallelism_threads=num_cpu,
                intra_op_parallelism_threads=num_cpu)
            config.gpu_options.allow_growth = True

        if make_default:
            return tf.InteractiveSession(config=config, graph=graph)
        else:
            return tf.Session(config=config, graph=graph)

class Runner(AbstractEnvRunner):
    '''
    Runner class used to make samples using policy for T timesteps
    This is the first part of PPO algorithm.
    '''

    def __init__(self, env, model, nsteps, gamma, lam):
        super(Runner, self).__init__(env=env, model=model, nsteps=nsteps)
        self.lam = lam
        self.gamma = gamma
        #self.env.set_model(self.model)

    def sf01(self, arr):
        """
        swap and then flatten axes 0 and 1
        """
        return arr
        s = arr.shape
        return arr.swapaxes(0, 1).reshape(s[0] * s[1], *s[2:])
    def run(self):
        # 20201009 ID might not be needed for single robot learning
        # 20201017 Resize all the array as single row)
        mb_ids, mb_obs, mb_rewards, mb_actions, mb_values, mb_dones, mb_neglogpacs = [],[],[],[],[],[],[]
        mb_states = self.states
        epinfos = []
        self.ids, self.obs = self.env.reset()
        self.dones = [False] * self.env.num_robots
        for _ in range(self.nsteps):
            #print self.obs
            mb_ids.append(self.ids)
            actions, values, self.states, neglogpacs = self.model.step(self.obs, self.states, self.dones)
            mb_obs.append(self.obs)
            mb_actions.append(actions)
            mb_values.append(values)
            mb_neglogpacs.append(neglogpacs)
            mb_dones.append(self.dones)

            # 20201009 Need to modify these inputs
            self.ids, self.obs, rewards, self.dones, infos = self.env.step(actions, 0.1)

            for info in infos:
                maybeepinfo = info.get('episode')
                if maybeepinfo: epinfos.append(maybeepinfo)
            mb_rewards.append(rewards)
        #batch of steps to batch of rollouts
        #mb_obs = np.asarray(mb_obs, dtype=self.obs.dtype)
        mb_ids = np.asarray(mb_ids)
        mb_rewards = np.asarray(mb_rewards, dtype=np.float32)
        mb_actions = np.asarray(mb_actions)
        mb_values = np.asarray(mb_values, dtype=np.float32)
        mb_neglogpacs = np.asarray(mb_neglogpacs, dtype=np.float32)
        mb_dones = np.asarray(mb_dones, dtype=np.bool)
        last_values = self.model.value(self.obs, self.states, self.dones)
        #discount/bootstrap off value fn
        mb_returns = np.zeros_like(mb_rewards)
        mb_advs = np.zeros_like(mb_rewards)
        lastgaelam = 0
        for t in reversed(range(self.nsteps)):
            if t == self.nsteps - 1:
                nextnonterminal = 1.0 #- self.dones
                nextvalues = last_values
            else:
                nextnonterminal = 1.0 #- mb_dones[t+1]
                nextvalues = mb_values[t+1]
            delta = mb_rewards[t] + self.gamma * nextvalues * nextnonterminal - mb_values[t]
            mb_advs[t] = lastgaelam = delta + self.gamma * self.lam * nextnonterminal * lastgaelam
        mb_returns = mb_advs + mb_values
        return mb_ids, mb_obs, self.sf01(mb_returns), self.sf01(mb_dones), self.sf01(mb_actions), self.sf01(mb_values), self.sf01(mb_neglogpacs), \
                mb_states, epinfos

class PPO:
    '''
    The main PPO class. The whole PPO algorithm is executed in 'learn' function
    '''

    def __init__(self, policy, env, nsteps, total_timesteps, ent_coef, lr,
          vf_coef=0.5,  max_grad_norm=0.5, gamma=0.99, lam=0.95,
          log_interval=10, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=20, restore_path=None, deterministic=False):

        if isinstance(lr, float): lr = self.constfn(lr)
        else: assert callable(lr)
        if isinstance(cliprange, float): cliprange = self.constfn(cliprange)
        else: assert callable(cliprange)
        self.policy = policy
        self.env = env
        self.nsteps = nsteps
        self.total_timesteps = total_timesteps
        self.ent_coef = ent_coef
        self.lr = lr
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.lam = lam
        self.log_interval = log_interval
        self.nminibatches = nminibatches
        self.noptepochs = noptepochs
        self.cliprange = cliprange
        self.save_interval = save_interval
        self.restore_path = restore_path
        self.deterministic = deterministic
    
    def constfn(self, val):
        def f(_):
            return val
        return f

    def learn(self):
        # For logging
        time_str = time.strftime("%Y%m%d-%H%M%S")
        logger_ins = logger.Logger('/home/swn/catkin_ws/src/turtlebot3_waypoint_navigation/src/log', output_formats=[logger.HumanOutputFormat(sys.stdout)])
        board_logger = tensorboard_logging.Logger(os.path.join(logger_ins.get_dir(), "tf_board", time_str))

        # reassigning the members of class into this function for simplicity
        total_timesteps = int(self.total_timesteps)
        nenvs = 1
        #nenvs = env.num_envs # for multiple instance training
        ob_space = self.env.observation_space
        ac_space = self.env.action_space
        nbatch = nenvs * self.nsteps
        nminibatches = self.nminibatches
        nbatch_train = nbatch // nminibatches
        noptepochs = self.noptepochs
        nsteps = self.nsteps
        save_interval = self.save_interval
        log_interval = self.log_interval
        restore_path = self.restore_path
        gamma = self.gamma
        lam = self.lam
        lr = self.lr
        cliprange = self.cliprange
        deterministic = self.deterministic

        # Define a function to make Actor-Critic Model 
        make_model = lambda : Model(policy=self.policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nenvs, nbatch_train=nbatch_train,
                                    nsteps=self.nsteps, ent_coef=self.ent_coef, vf_coef=self.vf_coef,
                                    max_grad_norm=self.max_grad_norm, deterministic=self.deterministic)
        
        # Save function 
        # if save_interval and logger_ins.get_dir():
        #     import cloudpickle
        #     with open(osp.join(logger_ins.get_dir(), 'make_model.pkl'), 'wb') as fh:
        #         fh.write(cloudpickle.dumps(make_model))

        # Make a model
        model = make_model()

        # Restore when the path is provided
        if restore_path is not None:
            model.restore(restore_path)
        
        # Create a runner instance (generating samples with nsteps)
        runner = Runner(env=self.env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

        # Double ended queue with max size 100 to store episode info
        epinfobuf = deque(maxlen=100)

        # Get the start time
        tfirststart = time.time()

        # Calculate # for update (iteration)
        nupdates = total_timesteps//nbatch
        assert(nupdates > 0)

        '''
        PPO (iterating)
        1. Run policy in the environment for T timesteps
        2. Compute advantage estimates (in Model class)
        3. Optimise Loss w.r.t weights of policy, with K epochs and minibatch size M < N (# of actors) * T (timesteps)
        4. Update weights (in Model class)
        '''
        # In every update, one loop of PPO algorithm is executed
        for update in range(1, nupdates+1):
            
            # INITIALISE PARAMETERS
            assert nbatch % nminibatches == 0
            nbatch_train = nbatch // nminibatches
            tstart = time.time()
            frac = 1.0 - (update - 1.0) / nupdates
            lrnow = lr(frac)
            cliprangenow = cliprange(frac)

            # 1. Run policy and get samples for nsteps
            ids, obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()
            print("ids {}, obs {}, returns {}, masks {}, actions {}, values {}, neglogpacs {}, states {}, epinfos {}".format(type(ids), type(obs), type(returns), type(masks), type(actions), type(values), type(neglogpacs), type(states), type(epinfos)))
            #print("obs: {}, actions: {}".format(np.asarray(obs).shape, np.asarray(actions).shape))
            epinfobuf.extend(epinfos)
            mblossvals = []

            # Do not train or log if in deterministic mode:
            if deterministic:
                continue
            
            # 3. Optimise Loss w.r.t weights of policy, with K epochs and minibatch size M < N (# of actors) * T (timesteps)
            if states is None: # nonrecurrent version
                #
                inds = np.arange(nbatch)
                # Update weights using optimiser by noptepochs
                for _ in range(noptepochs):
                    #np.random.shuffle(inds)

                    # In each epoch, update weights using samples every minibatch in the total batch
                    # epoch = m(32)*minibatch(4)
                    for start in range(0, nbatch, nbatch_train):
                        end = start + nbatch_train
                        mbinds = inds[start:end]
                        print("mbind: {}".format(mbinds))
                        returns_np = np.asarray(returns[mbinds])
                        # 4. Update weights
                        mblossvals.append(model.train(lrnow, cliprangenow, ids[mbinds], [obs[i] for i in mbinds], returns[mbinds],
                                        masks[mbinds], actions[mbinds], values[mbinds],
                                        neglogpacs[mbinds]))

            else: # recurrent version
                assert nenvs % nminibatches == 0
                envsperbatch = nenvs // nminibatches
                envinds = np.arange(nenvs)
                flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
                envsperbatch = nbatch_train // nsteps
                for _ in range(noptepochs):
                    #np.random.shuffle(envinds)
                    for start in range(0, nenvs, envsperbatch):
                        end = start + envsperbatch
                        mbenvinds = envinds[start:end]
                        mbflatinds = flatinds[mbenvinds].ravel()
                        print(mbflatinds)
                        slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                        mbstates = states[mbenvinds]
                        mblossvals.append(model.train(lrnow, cliprangenow,
                                        [obs[i] for i in mbinds], returns[mbflatinds], masks[mbflatinds], actions[mbflatinds],
                                        values[mbflatinds], neglogpacs[mbflatinds], mbstates))

            # Calculate mean loss
            lossvals = np.mean(mblossvals, axis=0)

            tnow = time.time()
            fps = int(nbatch / (tnow - tstart))
            
            '''
            Logging and saving model & weights
            '''
            if update % log_interval == 0 or update == 1:
                #ev = explained_variance(values, returns)
                logger_ins.logkv("serial_timesteps", update*nsteps)
                logger_ins.logkv("nupdates", update)
                logger_ins.logkv("total_timesteps", update*nbatch)
                logger_ins.logkv("fps", fps)
                #logger.logkv("explained_variance", float(ev))
                logger_ins.logkv('eprewmean', self.safemean([epinfo['r'] for epinfo in epinfobuf]))
                logger_ins.logkv('eplenmean', self.safemean([epinfo['l'] for epinfo in epinfobuf]))
                logger_ins.logkv('time_elapsed', tnow - tfirststart)
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    logger_ins.logkv(lossname, lossval)
                logger_ins.dumpkvs()
                for (lossval, lossname) in zip(lossvals, model.loss_names):
                    board_logger.log_scalar(lossname, lossval, update)
                board_logger.log_scalar("eprewmean", self.safemean([epinfo['r'] for epinfo in epinfobuf]), update)
                board_logger.flush()
            if save_interval and (update % save_interval == 0 or update == 1) and logger_ins.get_dir():
                checkdir = osp.join(logger_ins.get_dir(), 'checkpoints')
                if not os.path.isdir(checkdir):
                    os.makedirs(checkdir)
                savepath = osp.join(checkdir, '%.5i'%update +"r"+"{:.2f}".format(self.safemean([epinfo['r'] for epinfo in epinfobuf])))
                print('Saving to', savepath)
                model.save(savepath)
        print("Done with training. Exiting.")
        self.env.close()
        return model

    def safemean(self, xs):
       return np.nan if len(xs) == 0 else np.mean(xs)

def main():
    env = phero_turtlebot_turtlebot3_repellent_ppo_multi.Env()
    PPO_a = PPO(policy=PheroTurtlebotPolicy, env=env, nsteps=256, nminibatches=4, lam=0.95, gamma=0.99,
                noptepochs=10, log_interval=10, ent_coef=.01,
                lr=lambda f: f* 5.5e-4,
                cliprange=lambda f: f*0.3,
                total_timesteps=5000000,
                deterministic=False)
    PPO_a.learn()

if __name__ == '__main__':
    main()