
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_agents as tfa
import reverb

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.drivers import driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import py_uniform_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import general.utility as u
import general.plotting as gpl
import disentangled.aux as da
import disentangled.regularizer as dr
import disentangled.disentanglers as dd

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfspec = tfa.specs
tfaa = tfa.agents
tfa_ddpg = tfa.agents.ddpg

def make_environment(*args, duration=100, convert_tf=True, **kwargs):
    py_env = RLEnvironment(*args, **kwargs)
    # new_env = tfa.environments.TimeLimit(new_env, duration)
    if convert_tf:
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
        out = (py_env, tf_env)
    else:
        py_env
    return out

class RLEnvironment(tfa.environments.PyEnvironment):

    def __init__(self, dg, n_tasks, *args, discount=.9, eps=.6, f_type=np.float32,
                 i_type=np.int32, episode_length=100, n_tasks_per_samp=None,
                 reward_pos_weight=1, reward_neg_weight=1, **task_kwargs):
        self.n_tasks = n_tasks
        if n_tasks_per_samp is None:
            self.n_tasks_per_samp = n_tasks
        else:
            self.n_tasks_per_samp = n_tasks_per_samp
        self.rng = np.random.default_rng()
        self.episode_length = episode_length
        self.eps = eps
        self.f_type = f_type
        self.i_type = i_type
        self.discount = np.array(discount, dtype=f_type)
        out = dd.make_tasks(dg.input_dim, n_tasks, **task_kwargs)
        self.tasks, _, _ = out
        self.dg = dg

        self.reward_pos_weight = reward_pos_weight
        self.reward_neg_weight = reward_neg_weight

        self._observation_spec = tfspec.ArraySpec((dg.output_dim,), f_type)

        self._reward_spec = tfspec.ArraySpec((), f_type)
        self._discount_spec = tfspec.ArraySpec((), f_type)
        self._step_type_spec = tfspec.ArraySpec((), i_type)
        self._action_spec = tfspec.BoundedArraySpec((len(self.tasks),), f_type,
                                                     minimum=-1, maximum=1)
        self._time_step_spec = tfa.trajectories.TimeStep(
            step_type=self._step_type_spec, observation=self._observation_spec,
            reward=self._reward_spec, discount=self._discount_spec)

    def _reset(self):
        self.step_number = 0
        initial_samp, initial_rep = self.dg.sample_reps(1)
        self.current_samp = initial_samp
        initial_rep = np.squeeze(initial_rep).astype(self.f_type)
        step_type = np.array(0, dtype=self.i_type) 
        reward = np.array(0., dtype=self.f_type)
        discount = self.discount
        current_step = tfa.trajectories.TimeStep(step_type=step_type,
                                                 observation=initial_rep,
                                                 reward=reward,
                                                 discount=discount)
        return current_step

    def _choose_task(self):
        ind = self.rng.choice(self.n_tasks, size=self.n_tasks_per_samp,
                              replace=False)
        return ind
    
    def compute_reward(self, action, no_flatten=False):
        action = np.squeeze(action)
        mask_one = action > 1 - self.eps
        mask_zero = action <= -1 + self.eps

        corr_list = list(t(self.current_samp) for t in self.tasks)
        corr_vec = np.squeeze(np.array(corr_list).astype(bool))

        reward_vec = np.logical_or(mask_one * corr_vec,
                                   mask_zero * np.logical_not(corr_vec))
        neg_reward_vec = np.logical_or(mask_one * np.logical_not(corr_vec),
                                       mask_zero * corr_vec)
        reward_vec = (self.reward_pos_weight*reward_vec.astype(self.f_type)
                      - self.reward_neg_weight*neg_reward_vec.astype(self.f_type))
        if len(self.tasks) > 1:
            reward_vec = reward_vec[self._choose_task()]
        else:
            reward_vec = np.expand_dims(reward_vec, 0).astype(self.f_type)
        if no_flatten:
            reward = reward_vec
        else:
            reward = np.sum(reward_vec)
        return reward
    
    def _step(self, action):
        reward = self.compute_reward(action)
        self.current_samp, observation = self.dg.sample_reps(1)
        discount = self.discount
        if self.step_number < self.episode_length:
            step_type = np.array(1, dtype=self.i_type)
        else:
            step_type = np.array(2, dtype=self.i_type)
        self.step_number += 1
        observation = np.squeeze(observation).astype(self.f_type)
        next_step = tfa.trajectories.TimeStep(step_type=step_type,
                                              observation=observation,
                                              reward=reward,
                                              discount=discount)
        return next_step

    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def time_step_spec(self):
        return self._time_step_spec

    def step_type_spec(self):
        return self._step_type_spec


def compute_avg_return(environment, policy, num_episodes=10):

    total_return = 0.0
    for _ in range(num_episodes):
        
        time_step = environment.reset()
        episode_return = 0.0
        episode_steps = 0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            # print(environment.compute_reward(action_step))
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            episode_steps += 1
    
        total_return += episode_return / episode_steps
        
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


class SimpleActorNetwork(tfa_ddpg.actor_network.ActorNetwork):

    def __init__(self, input_spec, output_spec, layer_sizes,
                 encoded_size, last_kernel_initializer=None,
                 act_func=tf.nn.relu, **layer_params):
        super().__init__(input_spec, output_spec, fc_layer_params=layer_sizes,
                         activation_fn=act_func,
                         last_kernel_initializer=last_kernel_initializer)
        
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=input_spec.shape))

        for ls in layer_sizes:
            l_i = tfkl.Dense(ls, activation=act_func, **layer_params)
            layer_list.append(l_i)

        layer_list.append(tfkl.Dense(encoded_size, activation=None))

        out_layer = tfkl.Dense(self._single_action_spec.shape.num_elements(),
                               activation=tf.nn.tanh,
                               kernel_initializer=last_kernel_initializer,
                               name='action')
        layer_list.append(out_layer)
        self._mlp_layers = layer_list

    def get_representation(self, inputs):
        return self.rep_model(inputs)


class RLDisentangler(dd.FlexibleDisentanglerAE):
    """ 
    The environment is samples from a DataGenerator
    An action is a vector of length <number of tasks> giving categorization
    Reward is given for each correct categorization
    """
    
    def __init__(self, env, actor_layers,
                 critic_action_layers, critic_obs_layers,
                 critic_joint_layers, train_sequence_length=2,
                 encoded_size=50, use_simple_actor=True):
        self.env = env
        self.train_sequence_length = train_sequence_length
        # self.input_dim = self.env.dg.input_dim
        action_spec = self.env.action_spec()
        observation_spec = self.env.observation_spec()
        # step_type_spec = self.env.step_type_spec()
        time_step_spec = self.env.time_step_spec()
        last_init = tf.random_uniform_initializer(minval=-0.003,
                                                  maxval=0.003)
        if use_simple_actor:
            self.actor = SimpleActorNetwork(
                observation_spec,
                action_spec, actor_layers, encoded_size,
                last_kernel_initializer=last_init)
        else:
            self.actor = tfa_ddpg.actor_network.ActorNetwork(
                observation_spec, action_spec,
                fc_layer_params=actor_layers,
                last_kernel_initializer=last_init)
        self.critic = tfa_ddpg.critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=critic_obs_layers,
            action_fc_layer_params=critic_action_layers,
            joint_fc_layer_params=critic_joint_layers)
        self.compiled = False

    def _compile(self, act_opt=None, crit_opt=None, actor_learning_rate=1e-4,
                 critic_learning_rate=2e-4, ou_stddev=.5, ou_damping=.2,
                 discount=.9):
        if act_opt is None:
            act_opt = tf.keras.optimizers.Adam(
                learning_rate=actor_learning_rate)
        if crit_opt is None:
            crit_opt = tf.keras.optimizers.Adam(
                learning_rate=actor_learning_rate)
        self.agent = tfaa.DdpgAgent(
            self.env.time_step_spec(), self.env.action_spec(),
            actor_network=self.actor, actor_optimizer=act_opt,
            critic_network=self.critic, critic_optimizer=crit_opt,
            ou_stddev=ou_stddev, ou_damping=ou_damping,
            gamma=discount)
        self.agent.initialize()
        self.compiled = True

    def get_representation(self, samples):
        layers = self.actor.layers[:-1]
        output = layers[0](samples)
        for layer in layers[1:]:
            output = layer(output)
        return output
        
    def fit_py(self, env=None, num_iterations=10000, initial_collect_steps=100,
               collect_steps_per_iteration=1, replay_buffer_max_length=100000,
               batch_size=64, log_interval=200, num_eval_episodes=10,
               eval_interval=1000, learning_rate=1e-3):
        if env is None:
            env = self.env
        if not self.compiled:
            self._compile()
        
        train_step_counter = tf.Variable(0)

        table_name = 'uniform_table'
        replay_buffer_signature = tensor_spec.from_spec(
            self.agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(
            replay_buffer_signature)

        random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                        env.action_spec())

        table = reverb.Table(
            table_name,
            max_size=replay_buffer_max_length,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=replay_buffer_signature)
            
        reverb_server = reverb.Server([table])
            
        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            self.agent.collect_data_spec,
            table_name=table_name,
            sequence_length=2,
            local_server=reverb_server)
            
        replay_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.py_client,
            table_name,
            sequence_length=2)

        ## NEED TO MAKE TF vs PY AGREE -- leading to issues
        py_driver.PyDriver(
            py_env, random_policy,
            py_tf_eager_policy.PyTFEagerPolicy(
                random_policy, use_tf_function=True),
            [replay_observer],
            max_steps=initial_collect_steps).run(py_env.reset())

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)

        iterator = iter(dataset)

        time_step = py_env.reset()
        collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            self.agent.collect_policy, use_tf_function=True)

        collect_driver = py_driver.PyDriver(
            py_env, collect_policy, [replay_observer],
            max_steps=collect_steps_per_iteration)

        # return collect_driver, iterator, dataset, replay_buffer, # reverb_server 
        returns = []
        print('entering for loop')
        for i in range(num_iterations):
            time_step, _ = collect_driver.run(time_step)
            experience, unused_info = next(iterator)
            train_loss = self.agent.train(experience).loss
            
            step = self.agent.train_step_counter.numpy()

            if step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, train_loss))

            if step % eval_interval == 0:
                avg_return = compute_avg_return(eval_env, self.agent.policy,
                                                num_eval_episodes)
                s = 'step = {0}: Average Return = {1}'.format(step, avg_return)
                print(s)
                returns.append(avg_return)
        return returns
    
    def fit_tf(self, env=None, num_iterations=10000, initial_collect_episodes=100,
               collect_episodes_per_iteration=1, replay_buffer_max_length=100000,
               batch_size=64, log_interval=200, num_eval_episodes=10,
               eval_interval=1000, learning_rate=1e-3, test_rep=None):
        if env is None:
            env = self.env
        if not self.compiled:
            self._compile()
        
        replay_buffer_signature = tensor_spec.from_spec(
            self.agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(
            replay_buffer_signature)
        random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                        env.action_spec())
        
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec, batch_size=env.batch_size,
            max_length=replay_buffer_max_length)
        
        replay_observer = replay_buffer.add_batch

        dynamic_episode_driver.DynamicEpisodeDriver(
            env, random_policy,
            [replay_observer],
            num_episodes=initial_collect_episodes).run(env.reset())

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3, num_steps=self.train_sequence_length,
            sample_batch_size=batch_size).prefetch(3)

        iterator = iter(dataset)

        time_step = env.reset()
        collect_policy = self.agent.collect_policy

        collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
            env, collect_policy, [replay_observer],
            num_episodes=collect_episodes_per_iteration)

        self.agent.train_step_counter.assign(0)
        returns = []
        print(replay_buffer.num_frames())
        if test_rep is not None:
            print(self.actor(test_rep))
        for i in range(num_iterations):
            time_step, _ = collect_driver.run(time_step)
            # print(time_step)
            # print(self.agent.collect_policy.action(time_step))
            experience, unused_info = next(iterator)
            train_loss = self.agent.train(experience).loss
            # train_actor_loss = self.agent.actor_loss(experience)
            train_actor_loss = 0
            train_critic_loss = 0
            # train_critic_loss = self.agent.critic_loss(experience)
            step = self.agent.train_step_counter.numpy()
            if step % log_interval == 0:
                if test_rep is not None:
                    print(self.actor(test_rep))
                s = 'step = {0}: loss = {1}, actor = {2}, critic = {3}'
                print(s.format(step, train_loss, train_actor_loss,
                               train_critic_loss))

            if step % eval_interval == 0:
                avg_return = compute_avg_return(env, self.agent.policy,
                                                num_eval_episodes)
                s = 'step = {0}: Average Return = {1}'.format(step, avg_return)
                print(s)
                returns.append(avg_return)
        return returns
