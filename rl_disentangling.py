import tensorflow as tf
import tensorflow_probability as tfp
import tf_agents as tfa
import reverb

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import py_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import reverb_replay_buffer
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


class RLEnvironment(tfa.environments.PyEnvironment):

    def __init__(self, dg, n_tasks, *args, discount=.9, eps=.1, **task_kwargs):
        self.discount = discount
        self.tasks = dd.make_tasks(dg.input_dim, n_tasks, **task_kwargs)
        self.dg = dg
        self._observation_spec = tfspec.TensorSpec((dg.output_dim,), float)
        self._reward_spec = tfspec.BoundedTensorSpec((len(self.tasks),), float,
                                                     minimum=0, maximum=1)
        self._discount_spec = tfspec.TensorSpec((1,), float)
        self._step_type_spec = tfspec.TensorSpec((1,), float)
        self._action_spec = tfspec.BoundedTensorSpec((len(self.tasks),), float,
                                                     minimum=0, maximum=1)
        self.TimeStep = c.namedtuple('TimeStep',
                                     ('step_type', 'reward', 'discount',
                                      'observation'))
        self._time_step_spec = tfa.trajectories.TimeStep(
            step_type=self._step_type_spec, observation=self._observation_spec,
            reward=self._reward_spec, discount=self._discount_spec)

    def _reset(self):
        initial_samp, initial_rep = self.dg.sample_reps(1)
        self.current_samp = initial_samp
        step_type = 0
        discount = self.discount
        reward = np.zeros(self.reward_spec().shape)
        current_step = self.TimeStep(step_type=step_type,
                                     observation=initial_rep,
                                     reward=reward,
                                     discount=discount)
        return current_step
        
    def _step(self, action):
        ctt = self.current_time_step()
        mask_one = action > 1 - eps
        mask_zero = action < eps
        corr_vec = np.array(list(t(self.current_samp) for t in self.tasks))
        reward = np.logical_or(mask_one == corr_vec,
                               mask_zero == corr_vec)
        self.current_samp, observation = self.dg.sample_reps(1)
        discount = self.discount
        step_type = 0
        next_step = self.TimeStep(step_type=step_type,
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


class RLDisentangler(dd.FlexibleDisentanglerAE):
    """ 
    The environment is samples from a DataGenerator
    An action is a vector of length <number of tasks> giving categorization
    Reward is given for each correct categorization
    """

    def __init__(self, env, actor_layers,
                 critic_action_layers, critic_obs_layers,
                 critic_joint_layers):
        self.env = env
        self.input_dim = self.env.dg.input_dim
        action_spec = self.env.action_spec()
        observation_spec = self.env.observation_spec()
        step_type_spec = self.env.step_type_spec()
        time_step_spec = self.env.time_step_spec()
        self.actor = tfa_ddpg.actor_network.ActorNetwork(
            observation_spec,
            action_spec,
            fc_layer_params=actor_layers)
        self.critic = tfa_ddpg.critic_network.CriticNetwork(
            (observation_spec, action_spec),
            observation_fc_layer_params=critic_obs_layers,
            action_fc_layer_params=critic_action_layers,
            joint_fc_layer_params=critic_joint_layers)
        self.compiled = False

    def _compile(self, act_opt=None, crit_opt=None, learning_rate=1e-3):
        if act_opt is None:
            act_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        if crit_opt is None:
            crit_opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.agent = tfaa.DdpgAgent(
            self.env.time_step_spec(), self.env.action_spec(),
            actor_network=self.actor, actor_optimizer=act_opt,
            critic_network=self.critic, critic_optimizer=crit_opt)
        self.agent.initialize()
        self.compiled = True

        

    ## WHAT IS BEST WAY TO IMPLEMENT MULTI-TASK Q? TRY DDPG
    def fit(self, env, num_iterations=10000, initial_collect_steps=100,
            collect_steps_per_iteration=1, replay_buffer_max_length=100000,
            batch_size=64, log_interval=200, num_eval_episodes=10,
            eval_interval=1000, learning_rate=1e-3):
        if not self.compiled:
            self._compile()
        
        train_step_counter = tf.Variable(0)

        table_name = 'uniform_table'
        replay_buffer_signature = tensor_spec.from_spec(
            self.agent.collect_data_spec)
        replay_buffer_signature = tensor_spec.add_outer_dim(
            replay_buffer_signature)

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

        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
            replay_buffer.py_client,
            table_name,
            sequence_length=2)

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)
        
        iterator = iter(dataset)

        time_step = env.reset()

        collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
            self.agent.collect_policy, use_tf_function=True)
            
        collect_driver = py_driver.PyDriver(
            env, collect_policy, [rb_observer],
            max_steps=collect_steps_per_iteration)
        returns = []
        for _ in range(num_iterations):
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
    
