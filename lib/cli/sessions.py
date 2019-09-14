# UTILITY LIBRARIES
import pickle
import time
import numpy as np
import re
import sys
import pandas as pd
import tensorflow as tf
import optuna

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

# ENVIRONMENT
from lib.envs.simulated import SimulatedEnv
from lib.envs.live import LiveEnv # AGENTS

# UTILS
from lib.utils.added_tools import maybe_make_dir, dir_setup, generate_actions

# DEFINE GLOB VARIABLES
ACTIONS = ["SELL", "HOLD", "BUY"]

class Static_Session:
  def __init__(self, mode, weights, test_episodes, initial_invest, session_name):
    self.session_name = session_name
    self.mode = mode
    self.test_episodes = test_episodes
    self.initial_invest = initial_invest
    self.portfolio_value = []
    self.test_ep_rewards = []
    self.losses = 0
    self.actions = generate_actions()
    self.timestamp = dir_setup(mode)
    self.env = None

    data = pd.read_csv("./data/ta_all_mini_clean.csv")

    n_features = data.shape[1]

    train_data = data.iloc[:-10000, :]
    validation_data = data.iloc[-10000:-1000, :]
    test_data = data.iloc[-1000:, :]

    self.train_env = SimulatedEnv(train_data, self.initial_invest, self.mode)
    self.validation_env = SimulatedEnv(validation_data, self.initial_invest, self.mode)
    self.test_env = SimulatedEnv(test_data, self.initial_invest, self.mode)
    
    self.train_env = DummyVecEnv([lambda: self.train_env])
    self.validation_env = DummyVecEnv([lambda: self.validation_env])
    self.test_env = DummyVecEnv([lambda: self.test_env])

    state_size = n_features
    action_size = len(ACTIONS) * 10 

    self.f = open("stories/{}-{}-{}.csv".format(self.timestamp, self.mode, "BTC"),"w+")
    self.f.write("OPERATION,AMOUNT,STOCKS_OWNED,CASH_IN_HAND,PORTFOLIO_VALUE,OPEN_PRICE\n")
    
    self.optuna_study = optuna.create_study(
            study_name="test_study", 
            storage="sqlite:///data/params.db", 
            load_if_exists=True)
    
    if mode == "optimization":
      self.run_optimization()
    if mode == "train":
      self.run_train()
    # if mode == "validation":
    #   self.run_validation()

  def update_stats(self, e, episode_reward):
     # Calculate average gain
      self.test_ep_rewards.append(episode_reward)
      if not e % AGGREGATE_STATS_EVERY or e == 1:
        
        average_reward = sum(self.test_ep_rewards[-AGGREGATE_STATS_EVERY:])/len(self.test_ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(self.test_ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(self.test_ep_rewards[-AGGREGATE_STATS_EVERY:])       
  def print_stats(self, e, info):
    # Print stats
    print("episode: {}/{}, episode end value: {}".format(
      e + 1, self.test_episodes, info[0]['cur_val']))
    print("TOTAL LOSSES: {}".format(self.losses))
  def save_weights(self, e):
    if self.mode == 'train' and (e + 1) % 10 == 0:  # checkpoint weights
        agent.save('weights/{}-{}.h5'.format(self.timestamp, 'BTC'))    
  def write_to_story(self, action, info):
    self.f.write("{},{},{},{},{},{}\n".format(ACTIONS[self.actions[action][0]],self.actions[action][1], info['owned_stocks'], info['cash_in_hand'], info['cur_val'], info['price']))
  
  def optimize_agent_params(self, trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 16, 2048)),
        'gamma': trial.suggest_loguniform('gamma', 0.9, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1.),
        'ent_coef': trial.suggest_loguniform('ent_coef', 1e-8, 1e-1),
        'cliprange': trial.suggest_uniform('cliprange', 0.1, 0.4),
        'noptepochs': int(trial.suggest_loguniform('noptepochs', 1, 48)),
        'lam': trial.suggest_uniform('lam', 0.8, 1.)
    }
  def optimize_params(self, trial, n_prune_evals_per_trial: int = 2, n_tests_per_eval: int = 1):
    model_params = self.optimize_agent_params(trial)
    model = PPO2(MlpPolicy, 
                self.train_env, 
                verbose=1, 
                tensorboard_log="./logs/", 
                **model_params)

    n_steps_per_eval = 5000000

    for eval_idx in range(n_prune_evals_per_trial):
        try:
            model.learn(n_steps_per_eval)
        except AssertionError:
            raise
 
        last_reward = self.run_test(model)
        trial.report(-1 * last_reward, eval_idx)

        if trial.should_prune(eval_idx):
            raise optuna.structs.TrialPruned()

    return -1 * last_reward
  def get_model_params(self):
    params = self.optuna_study.best_trial.params
    return {
        'n_steps': int(params['n_steps']),
        'gamma': params['gamma'],
        'learning_rate': params['learning_rate'],
        'ent_coef': params['ent_coef'],
        'cliprange': params['cliprange'],
        'noptepochs': int(params['noptepochs']),
        'lam': params['lam'],
    }
  
  def run_optimization(self, n_trials: int = 20):
    try:
      self.optuna_study.optimize(self.optimize_params, n_trials=n_trials, n_jobs=1)
    except KeyboardInterrupt:
      pass
    return self.optuna_study.trials_dataframe()
  def run_test(self, agent, validation = True):
    if validation: 
      env = self.validation_env
    else:
      env = self.test_env

    for e in range(self.test_episodes):
      # Reset the environment at every episode. 
      state = env.reset()      
      # Initialize variable to get reward stats. 
      episode_reward = []
      
      for time in range(0, 500):
        action, _states = agent.predict(state)
        next_state, reward, done, info = env.step(action)
        
        self.write_to_story(action[0], info[0])
        episode_reward.append(reward)
        state = next_state
        
        if done:
          if info[0]['cur_val'] < self.initial_invest:
            self.losses = self.losses + 1
          self.print_stats(e, info)
          self.f.write("{},{},{},{},{},{}\n".format(-1,-1,-1,-1,-1,-1))
          break
    return np.mean(episode_reward) 
  def run_train(self):
    # Save the agent
    model_params = self.get_model_params()
    model = PPO2(MlpPolicy, 
                self.train_env, 
                verbose=1, 
                tensorboard_log="./logs/", 
                **model_params)
    try:
      model.learn(total_timesteps=5000000)
      result = self.run_test(model)
      print("EPISODE_MEAN: {}".format(result))
      model.save(self.session_name)
    except KeyboardInterrupt:
      print("Saving model...")
      model.save(self.session_name)

# TODO fix
# def run_live_session(mode, weights, batch_size, test_episodes, initial_invest):

#   # Initialize session variables
#   portfolio_value, test_ep_rewards, losses = [], [], 0
  
#   #Setup initial directories and return a current timestamp
#   timestamp = dir_setup(mode)

#   # Download data and create an environment and corrispective statesize actionsize
#   env, state_size, action_size = live_env_setup(initial_invest, 63, 3)

#   # TODO implement a choice for DDQN / DQN
#   # Setup the DDQN agent
#   agent = DDQNAgent(state_size, action_size, mode)

#   # Create story for session
#   f = open("stories/{}-{}-{}.csv".format(timestamp, mode, "BTC"),"w+")
#   f.write("OPERATION,AMOUNT,CRYPTO_OWNED,CASH_IN_HAND,PORTFOLIO_VALUE,OPEN_PRICE\n")

#   timestamp, agent = test_setup(mode, weights, agent, timestamp)

#   state = env._reset()

#   for time in range(env.n_step):
#     # Act a consequence of next state
#     action = agent.act(state)

#     # Make a step and print the consequence. 
#     next_state, reward, info = env._step(action)
  
#     f.write("{},{},{},{},{},{}\n".format(\
#       ACTIONS[action], \
#       info['crypto_owned'], \
#       info['cash_in_hand'], \
#       info['cur_val'], \
#       info['price']))
    
#     state = next_state