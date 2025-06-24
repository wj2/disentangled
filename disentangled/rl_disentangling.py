
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tf_agents as tfa
# import reverb

from tf_agents.agents import tf_agent
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
# from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.replay_buffers import py_uniform_replay_buffer
# from tf_agents.replay_buffers import reverb_utils
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common
from tf_agents.utils import nest_utils

import general.utility as u
import general.plotting as gpl
import disentangled.aux as da
import disentangled.regularizer as dr
import disentangled.disentanglers as dd
import disentangled.characterization as dc
import collections

tfk = tf.keras
tfkl = tf.keras.layers
tfpl = tfp.layers
tfd = tfp.distributions
tfspec = tfa.specs
tfaa = tfa.agents
tfa_ddpg = tfa.agents.ddpg

def training_characterizing_script(envs, model_maker, ou_stddev=1,
                                   n_reps=10, initial_collects=None,
                                   model_n_bounds=(4, 4.5),
                                   model_n_diffs=2, batch_size=200,
                                   distr_type='normal', gpu_samples=False,
                                   eval_n_iters=10, p_mean=True, **fit_kwargs):
    if initial_collects is None:
        initial_collects = (1000, 5000, 10000)
    train_samples = np.logspace(model_n_bounds[0], model_n_bounds[1],
                                model_n_diffs, dtype=int)

    all_models = np.zeros((len(envs), len(initial_collects), model_n_diffs,
                           n_reps), dtype=object)
    all_hist = np.zeros_like(all_models)
    print(all_models.shape)
    for ind in u.make_array_ind_iterator(all_models.shape):
        m_ind = model_maker(envs[ind[0]][1])
        m_ind._compile(ou_stddev=ou_stddev)
        print(train_samples[ind[2]])
        hist = m_ind.fit_tf(envs[ind[0]][1],
                            num_iterations=train_samples[ind[2]],
                            batch_size=batch_size,
                            initial_collect_episodes=initial_collects[ind[1]],
                            **fit_kwargs)
        all_models[ind] = m_ind.actor
        print(hist)
        all_hist[ind] = hist

    dg_use = envs[0][0].dg
    models = all_models
    try:
        train_d2 = dg_use.source_distribution.make_partition()
    except AttributeError:
        if distr_type == 'normal':
            train_d2 = da.HalfMultidimensionalNormal.partition(
                dg_use.source_distribution)
        elif distr_type == 'uniform':
            train_d2 = da.HalfMultidimensionalUniform.partition(
                dg_use.source_distribution)
        else:
            raise IOError('distribution type indicated ({}) is not '
                          'recognized'.format(distr_type))
                
    train_ds = (None, train_d2)
    test_ds = (None, train_d2.flip())

    if gpu_samples:
        n_train_samples = 2*10**3
        n_test_samples = 10**3
        n_save_samps = int(n_save_samps/10)
    else:
        n_train_samples = 2*10**3
        n_test_samples = 10**3

    p, c = dc.evaluate_multiple_models_dims(
      dg_use, models, None, test_ds, train_distributions=train_ds,
      n_iters=eval_n_iters, n_train_samples=n_train_samples,
      n_test_samples=n_test_samples, mean=p_mean)

    lts_scores = dc.find_linear_mappings(
      dg_use, models, half=True, n_test_samps=n_test_samples)

    return dg_use, (models, all_hist), (p, c), lts_scores, None                   


class DdpgInfo(collections.namedtuple(
    'DdpgInfo', ('actor_loss', 'critic_loss'))):
  pass

def make_environment(*args, duration=100, convert_tf=True, **kwargs):
    py_env = RLEnvironment(*args, **kwargs)
    # new_env = tfa.environments.TimeLimit(new_env, duration)
    if convert_tf:
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
        out = (py_env, tf_env)
    else:
        out = py_env
    return out

class RLEnvironment(tfa.environments.PyEnvironment):

    def __init__(self, dg, n_tasks, *args, discount=0, eps=1, f_type=np.float32,
                 i_type=np.int32, episode_length=100, n_tasks_per_samp=None,
                 reward_pos_weight=1, reward_neg_weight=1, multi_reward=False,
                 include_task_mask=False, **task_kwargs):
        self.include_task_mask = include_task_mask
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
        out = dd.make_tasks(dg.input_dim, n_tasks, **task_kwargs)
        self.tasks, _, _ = out
        self.dg = dg

        self.reward_pos_weight = reward_pos_weight
        self.reward_neg_weight = reward_neg_weight

        if include_task_mask:
            observation_shape = (dg.output_dim + self.n_tasks,)
        else:
            observation_shape = (dg.output_dim,)

        self._observation_spec = tfspec.ArraySpec(observation_shape, f_type)

        self.multi_reward = multi_reward
        if self.multi_reward:
            rew_shape = (len(self.tasks),)
        else:
            rew_shape = ()
        self.discount = f_type(discount) # np.ones((), dtype=f_type)*discount
        
        self._reward_spec = tfspec.ArraySpec(rew_shape, f_type)
        self._discount_spec = tfspec.ArraySpec((), f_type)
        
        self._step_type_spec = tfspec.ArraySpec((), i_type)
        self._action_spec = tfspec.BoundedArraySpec((len(self.tasks),), f_type,
                                                     minimum=-1, maximum=1)
        self._time_step_spec = tfa.trajectories.TimeStep(
            step_type=self._step_type_spec, observation=self._observation_spec,
            reward=self._reward_spec, discount=self._discount_spec)

    def _get_observation(self, task_mask=None):
        initial_samp, initial_rep = self.dg.sample_reps(1)
        out_rep = np.squeeze(initial_rep)
        if task_mask is not None and self.include_task_mask:
            out_rep = np.concatenate((out_rep, task_mask),
                                     axis=0)
        out_rep = out_rep.astype(self.f_type)
        return initial_samp, out_rep

    def _reset(self):
        self.step_number = 0
        task_mask = self._choose_task()
        initial_samp, initial_rep = self._get_observation(task_mask)
        self.current_samp = initial_samp
        self.current_task_mask = task_mask
        step_type = np.array(0, dtype=self.i_type)
        reward = np.zeros(self.reward_spec().shape, dtype=self.f_type)
        discount = self.discount
        current_step = tfa.trajectories.TimeStep(step_type=step_type,
                                                 observation=initial_rep,
                                                 reward=reward,
                                                 discount=discount)
        return current_step

    def _choose_task(self):
        if self.include_task_mask:
            ind = self.rng.choice(self.n_tasks, size=self.n_tasks_per_samp,
                                  replace=False)
            task_mask = np.zeros(self.n_tasks, dtype=bool)
            task_mask[ind] = True
        else:
            task_mask = None
        return task_mask
    
    def compute_reward(self, action, no_flatten=False):
        if self.multi_reward:
            no_flatten = True
        action = np.squeeze(action)
        mask_one = action > 1 - self.eps
        mask_zero = action <= -1 + self.eps

        corr_list = list(t(self.current_samp) for t in self.tasks)
        corr_vec = np.squeeze(np.array(corr_list).astype(bool))
        if len(self.tasks) == 1:
            corr_vec = np.expand_dims(corr_vec, axis=0)

        reward_vec = np.logical_or(mask_one * corr_vec,
                                   mask_zero * np.logical_not(corr_vec))
        neg_reward_vec = np.logical_or(mask_one * np.logical_not(corr_vec),
                                       mask_zero * corr_vec)
        reward_vec = (self.reward_pos_weight*reward_vec.astype(self.f_type)
                      - self.reward_neg_weight*neg_reward_vec.astype(self.f_type))
        if self.current_task_mask is not None:
            reward_vec[np.logical_not(self.current_task_mask)] = 0
        else:
            reward_vec = np.expand_dims(reward_vec, 0).astype(self.f_type)
        if no_flatten:
            reward = reward_vec
        else:
            reward = np.sum(reward_vec)
        return reward
    
    def _step(self, action):
        reward = self.compute_reward(action)
        self.current_task_mask = self._choose_task()
        out = self._get_observation(self.current_task_mask)
        self.current_samp, observation = out
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

    def discount_spec(self):
        return self._discount_spec
    
    def reward_spec(self):
        return self._reward_spec
    
    def observation_spec(self):
        return self._observation_spec

    def action_spec(self):
        return self._action_spec

    def time_step_spec(self):
        return self._time_step_spec

    def step_type_spec(self):
        return self._step_type_spec


class DummyCriticNetwork():

    def __init__(self, *networks, mask_in_obs=True):
        self.networks = networks
        self.mask_in_obs = mask_in_obs

    def __call__(self, inp, mask=None, **kwargs):
        obs, action = inp
        if mask is None and not self.mask_in_obs:
            mask = tf.ones(action.shape, dtype=int)
        elif mask is None and self.mask_in_obs:
            obs_len = obs.shape[1] - action.shape[1]
            mask = obs[:, obs_len:]
            obs = obs[:, :obs_len]
        outs = tf.zeros(action.shape[0])
        for i, net in enumerate(self.networks):
            q_resp, other = net((obs, action[:, i:i+1]), **kwargs)
            outs = outs + q_resp*mask[:, i]
        return outs, other

    @property
    def variables(self):
        var_list = []
        for net in self.networks:
            var_list.extend(net.variables)
        return var_list

    @property
    def trainable_variables(self):
        var_list = []
        for net in self.networks:
            var_list.extend(net.trainable_variables)
        return var_list

class ExtendedDdpgAgent(tfaa.DdpgAgent):

    def __init__(self, time_step_spec, action_spec, actor_network,
                 critic_network, many_critics=False, target_critic_network=None,
                 **kwargs):
        if many_critics:
            cn = critic_network[0]
        else:
            cn = critic_network
        super().__init__(time_step_spec, action_spec, actor_network,
                         cn, target_critic_network=target_critic_network,
                         **kwargs)
        self.many_critics = many_critics
        if many_critics:
            n_tasks = action_spec.shape[0]
            single_action_spec = tfspec.BoundedTensorSpec(
                (1,), dtype=action_spec.dtype, minimum=action_spec.minimum,
                maximum=action_spec.maximum)
            self._many_critic_networks = critic_network
            critic_input_spec = (time_step_spec.observation, single_action_spec)
            list(cn.create_variables(critic_input_spec)
                 for cn in self._many_critic_networks)
            if target_critic_network:
                many_target_critic_networks = target_critic_network
                list(cn.create_variables(critic_input_spec)
                     for cn in many_target_critic_networks)
            else:
                many_target_critic_networks = (None,)*n_tasks
            tcns = []
            for i in range(n_tasks):
                tcn = common.maybe_copy_target_network_with_checks(
                    self._many_critic_networks[i],
                    many_target_critic_networks[i],
                    'TCN{}'.format(i),
                    input_spec=critic_input_spec)
                tcns.append(tcn)
            self._many_target_critic_networks = tcns
            self._target_critic_network = DummyCriticNetwork(*tcns)
            self._critic_network = DummyCriticNetwork(
              *self._many_critic_networks)

    def critic_loss(self,
                    time_steps,
                    actions,
                    next_time_steps,
                    weights=None,
                    training=False):
        """Computes the critic loss for DDPG training.

        Args:
        time_steps: A batch of timesteps.
        actions: A batch of actions.
        next_time_steps: A batch of next timesteps.
        weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.
        training: Whether this loss is being used for training.
        Returns:
        critic_loss: A scalar critic loss.
        """
        with tf.name_scope('critic_loss'):
            target_actions, _ = self._target_actor_network(
                next_time_steps.observation, step_type=next_time_steps.step_type,
                training=False)
            target_critic_net_input = (next_time_steps.observation, target_actions)
            target_q_values, _ = self._target_critic_network(
                target_critic_net_input, step_type=next_time_steps.step_type,
                training=False)

            td_targets = tf.stop_gradient(
                self._reward_scale_factor * next_time_steps.reward +
                self._gamma * next_time_steps.discount * target_q_values)
            
            critic_net_input = (time_steps.observation, actions)
            q_values, _ = self._critic_network(critic_net_input,
                                               step_type=time_steps.step_type,
                                               training=training)

            critic_loss = self._td_errors_loss_fn(td_targets, q_values)
            if nest_utils.is_batched_nested_tensors(
                    time_steps, self.time_step_spec, num_outer_dims=2):
                # Do a sum over the time dimension.
                critic_loss = tf.reduce_sum(critic_loss, axis=1)
            if weights is not None:
                critic_loss *= weights
            critic_loss = tf.reduce_mean(critic_loss)

            # c_mask = (np.array(q_values) < 0) == (np.array(td_targets) < 0)
            # pv = 10
            # action_mask = time_steps.observation[:pv, -2:]
            # print('acti', actions[:pv]*action_mask)
            # print('qval', q_values[:pv])
            # print('rwdv', td_targets[:pv])
            # print('diff', q_values[:pv] - td_targets[:pv])
            # print('corr', c_mask[:pv])
            # print('mcor', np.mean(c_mask))
            # print('loss', critic_loss)
            
            with tf.name_scope('Losses/'):
                tf.compat.v2.summary.scalar(
                    name='critic_loss', data=critic_loss, step=self.train_step_counter)

            if self._debug_summaries:
                td_errors = td_targets - q_values
                common.generate_tensor_summaries('td_errors', td_errors,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('td_targets', td_targets,
                                                 self.train_step_counter)
                common.generate_tensor_summaries('q_values', q_values,
                                                 self.train_step_counter)

        return critic_loss
    
    def actor_loss(self, time_steps, weights=None, training=False):
        """Computes the actor_loss for DDPG training.
        Args:
        time_steps: A batch of timesteps.
        weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.
        training: Whether this loss is being used for training.
        # TODO(b/124383618): Add an action norm regularizer.
        Returns:
        actor_loss: A scalar actor loss.
        """
        with tf.name_scope('actor_loss'):
            actions, _ = self._actor_network(time_steps.observation,
                                             step_type=time_steps.step_type,
                                             training=training)
            if self._actor_network.observation_includes_mask:
                obs_len = self._actor_network.observ_len
                action_mask = time_steps.observation[:, obs_len:] > 0
            else:
                action_mask = np.zeros_like(actions, dtype=bool)
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(actions)
                q_values, _ = self._critic_network((time_steps.observation, actions),
                                                   step_type=time_steps.step_type,
                                                   training=False)
                actions = tf.nest.flatten(actions)

            dqdas = tape.gradient([q_values], actions)
            # pv = 10
            # print('aqv', q_values[:pv])
            # print('act', actions[0][:pv])
            # print('dqd', dqdas[0][:pv])

            actor_losses = []
            for j, (dqda, action) in enumerate(zip(dqdas, actions)):
                if self._dqda_clipping is not None:
                    dqda = tf.clip_by_value(dqda, -1 * self._dqda_clipping,
                                            self._dqda_clipping)
                loss = common.element_wise_squared_loss(
                    tf.stop_gradient(dqda + action), action)
                # print('los', loss[:pv])
                
                if nest_utils.is_batched_nested_tensors(
                        time_steps, self.time_step_spec, num_outer_dims=2):
                    # Sum over the time dimension.
                    loss = tf.reduce_sum(loss, axis=1)
                if weights is not None:
                    loss *= weights
                # actor_losses.append(tf.Variable(0))
                # loss_vars = []
                for i in range(loss.shape[1]):
                    if j == 0:
                        actor_losses.append(0)
                    loss_masked = tf.boolean_mask(loss[:, i], action_mask[:, i])
                    actor_losses[i] = actor_losses[i] + tf.reduce_mean(loss_masked)
                # actor_losses.append(loss_vars)

            # actor_loss = tf.add_n(actor_losses)
            actor_loss = actor_losses
            # print(actor_loss)
            actor_loss_flat = tf.reduce_mean(actor_loss)

            with tf.name_scope('Losses/'):
                tf.compat.v2.summary.scalar(
                    name='actor_loss', data=actor_loss_flat,
                    step=self.train_step_counter)

        return actor_loss

    def _train(self, experience, weights=None):
        transition = self._as_transition(experience)
        time_steps, policy_steps, next_time_steps = transition
        actions = policy_steps.action

        # TODO(b/124382524): Apply a loss mask or filter boundary transitions.
        trainable_critic_variables = self._critic_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_critic_variables, ('No trainable critic variables to '
                                                'optimize.')
            tape.watch(trainable_critic_variables)
            critic_loss = self.critic_loss(time_steps, actions, next_time_steps,
                                           weights=weights, training=True)
        tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
        critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
        self._apply_gradients(critic_grads, trainable_critic_variables,
                              self._critic_optimizer)

        trainable_actor_variables = self._actor_network.trainable_variables
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            assert trainable_actor_variables, ('No trainable actor variables to '
                                               'optimize.')
            tape.watch(trainable_actor_variables)
            actor_loss = self.actor_loss(time_steps, weights=weights, training=True)
        tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
        # actor_loss = list(-al for al in actor_loss)
        actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
        self._apply_gradients(actor_grads, trainable_actor_variables,
                              self._actor_optimizer)

        self.train_step_counter.assign_add(1)
        self._update_target()

        # TODO(b/124382360): Compute per element TD loss and return in loss_info.
        total_loss = actor_loss + critic_loss
        return tf_agent.LossInfo(total_loss,
                                 DdpgInfo(actor_loss, critic_loss))
    
def compute_avg_return(environment, policy, num_episodes=10, py_env=None):
    if py_env is None:
        py_env = environment.pyenv._envs[0]
    if py_env is not None:
        total_return_pt = np.zeros(py_env.n_tasks)
    total_return = 0.0
    for _ in range(num_episodes):
        
        time_step = environment.reset()
        if py_env is not None:
            episode_return_pt = np.zeros(py_env.n_tasks)
        episode_return = 0.0
        
        episode_steps = 0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            if py_env is not None:
                rew_pt = py_env.compute_reward(action_step.action,
                                               no_flatten=True)
                episode_return_pt += rew_pt
            time_step = environment.step(action_step.action)
            rew = time_step.reward.numpy()[0]
            episode_return += rew
            episode_steps += 1
    
        total_return += episode_return / episode_steps
        if py_env is not None:
            total_return_pt += episode_return_pt / episode_steps
        
    avg_return = total_return / num_episodes
    out = avg_return
    if py_env is not None:
        avg_return_pt = total_return_pt / num_episodes
        out = (avg_return, avg_return_pt)
    return out

def nan_huber(y_true, y_pred):
    loss_fn = tf.keras.losses.Huber()
    y_pred_nans = tf.math.logical_not(tf.math.is_nan(y_pred))
    y_true_nans = tf.math.logical_not(tf.math.is_nan(y_true))
    yt = tf.boolean_mask(y_true, y_true_nans)
    yp = tf.boolean_mask(y_pred, y_pred_nans)
    out = loss_fn(yt, yp)
    return out    

class SimpleCriticNetwork(tfa_ddpg.critic_network.CriticNetwork):

    def __init__(self, input_spec, observation_len=None, **kwargs):
        if observation_len is not None:
            self.observ_len = observation_len
            self.observation_includes_mask = True
        super().__init__(input_spec, **kwargs)

    def call(self, inputs, *args, **kwargs):
        observations, actions = inputs
        if self.observation_includes_mask:
            obs = observations[:, :self.observ_len]
            mask = tf.cast(observations[:, self.observ_len:] > 0, float)

            actions = actions*mask
        else:
            obs = observations
            actions = actions
        return super().call((obs, actions), *args, **kwargs)

class SimpleActorNetwork(tfa_ddpg.actor_network.ActorNetwork):

    def __init__(self, input_spec, output_spec, layer_sizes,
                 encoded_size, last_kernel_initializer=None,
                 act_func=tf.nn.relu, observation_len=None,
                 **layer_params):
        if observation_len is not None:
            self.observ_len = observation_len
            self.observation_includes_mask = True
            input_shape = (self.observ_len,)
        else:
            input_shape = input_spec.shape
        super().__init__(input_spec, output_spec, fc_layer_params=layer_sizes,
                         activation_fn=act_func,
                         last_kernel_initializer=last_kernel_initializer)
        
        layer_list = []
        layer_list.append(tfkl.InputLayer(input_shape=input_shape))

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
        self.rep_model = None

    def call(self, observations, *args, repl_zero=True, **kwargs):
        if self.observation_includes_mask:
            obs = observations[:, :self.observ_len]
            mask = tf.cast(observations[:, self.observ_len:] > 0, tf.float32)
            mask = observations[:, self.observ_len:] > 0
        else:
            obs = observations
        resp, state = super().call(obs, *args, **kwargs)
        if self.observation_includes_mask:
            mult_mask = np.ones_like(mask, dtype=float)
            if repl_zero:
                repl_ = 0
            else:
                repl_ = np.nan
            mult_mask[np.logical_not(mask)] = repl_
            resp = resp*mult_mask
        return resp, state
        
    def get_representation(self, inputs):
        if self.rep_model is None:
            self.rep_model = tfk.Sequential(self._mlp_layers[:-1])
        return self.rep_model(inputs)
    
class RLDisentangler(dd.FlexibleDisentanglerAE):
    """ 
    The environment is samples from a DataGenerator
    An action is a vector of length <number of tasks> giving categorization
    Reward is given for each correct categorization
    """
    
    def __init__(self, env, actor_layers=(250, 150, 50),
                 critic_action_layers=(5,), critic_obs_layers=(5,),
                 critic_joint_layers=(5,), train_sequence_length=2,
                 encoded_size=50, use_simple_actor=True,
                 use_simple_critic=False,
                 observation_len=None, many_critics=False):
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
                last_kernel_initializer=last_init,
                observation_len=observation_len)
        else:
            self.actor = tfa_ddpg.actor_network.ActorNetwork(
                observation_spec, action_spec,
                fc_layer_params=actor_layers,
                last_kernel_initializer=last_init)
        self.many_critics = many_critics
        if use_simple_critic:
            self.critic = SimpleCriticNetwork(
                (observation_spec, action_spec),
                observation_fc_layer_params=critic_obs_layers,
                action_fc_layer_params=critic_action_layers,
                joint_fc_layer_params=critic_joint_layers,
                observation_len=observation_len)
        elif many_critics:
            self.critic = []
            n_tasks = action_spec.shape[0]
            single_action_spec = tfspec.BoundedTensorSpec(
                (1,), dtype=action_spec.dtype, minimum=action_spec.minimum,
                maximum=action_spec.maximum)
            if observation_len is not None:
                stripped_observation_spec = tfspec.TensorSpec(
                  (observation_len,), dtype=observation_spec.dtype)
            else:
                stripped_observation_spec = observation_spec
            for i in range(n_tasks):
                c_i = tfa_ddpg.critic_network.CriticNetwork(
                    (stripped_observation_spec, single_action_spec),
                    observation_fc_layer_params=critic_obs_layers,
                    action_fc_layer_params=critic_action_layers,
                    joint_fc_layer_params=critic_joint_layers)
                self.critic.append(c_i)
        else:
            self.critic = tfa_ddpg.critic_network.CriticNetwork(
                (observation_spec, action_spec),
                observation_fc_layer_params=critic_obs_layers,
                action_fc_layer_params=critic_action_layers,
                joint_fc_layer_params=critic_joint_layers)
        self.compiled = False

    def _compile(self, act_opt=None, crit_opt=None, actor_learning_rate=5e-4,
                 critic_learning_rate=2e-3, ou_stddev=1, ou_damping=.2,
                 discount=0, loss=None, target_update_tau=.5,
                 target_update_period=1):
        if act_opt is None:
            act_opt = tf.keras.optimizers.Adam(
                learning_rate=actor_learning_rate)
        if crit_opt is None:
            crit_opt = tf.keras.optimizers.Adam(
                learning_rate=critic_learning_rate)
        self.agent = ExtendedDdpgAgent(
            self.env.time_step_spec(), self.env.action_spec(),
            actor_network=self.actor, actor_optimizer=act_opt,
            critic_network=self.critic, critic_optimizer=crit_opt,
            ou_stddev=ou_stddev, ou_damping=ou_damping,
            gamma=discount, td_errors_loss_fn=loss,
            many_critics=self.many_critics,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period)
        self.agent.initialize()
        self.compiled = True

    def get_representation(self, samples):
        layers = self.actor.layers[:-1]
        output = layers[0](samples)
        for layer in layers[1:]:
            output = layer(output)
        return output
        
    def fit_tf(self, env=None, num_iterations=10000,
               initial_collect_episodes=1000,
               collect_episodes_per_iteration=1, replay_buffer_max_length=100000,
               batch_size=200, log_interval=200, num_eval_episodes=10,
               eval_interval=1000, learning_rate=1e-3, test_rep=None,
               append_returns=None, py_env=None):
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
        returns_pt = []
        if test_rep is not None:
            print(self.actor(test_rep))
        for i in range(num_iterations):
            time_step, _ = collect_driver.run(time_step)
            # print(time_step)
            # print(self.agent.collect_policy.action(time_step))
            experience, unused_info = next(iterator)
            all_loss = self.agent.train(experience)
            train_loss = all_loss.loss
            train_actor_loss = all_loss.extra.actor_loss
            train_critic_loss = all_loss.extra.critic_loss
            step = self.agent.train_step_counter.numpy()
            if step % log_interval == 0:
                if test_rep is not None:
                    print(self.actor(test_rep))
                s = 'step = {}: loss = {:0.4f}, actor = {:0.4f}, critic = {:0.4f}'
                print(s.format(step, np.mean(train_loss), np.mean(train_actor_loss),
                               np.mean(train_critic_loss)))
                
            if step % eval_interval == 0:
                out = compute_avg_return(env, self.agent.policy,
                                         num_eval_episodes,
                                         py_env=py_env)
                avg_return, avg_return_pt = out
                if append_returns is not None:
                    append_returns.append(avg_return)
                s = 'step = {}: Average Return = {:.2f} | '.format(step, avg_return)
                print(s, np.round(avg_return_pt, 2))
                returns.append(avg_return)
                returns_pt.append(avg_return_pt)
        return returns, returns_pt
