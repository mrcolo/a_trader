# UTILITY LIBRARIES
import pickle
import time
import numpy as np
import re
import sys
import pandas as pd
import tensorflow as tf
import optuna
import gym
from yahoo_fin import stock_info as si
import coloredlogs

from lib.utils.logger import init_logger
from lib.utils.generate_ta import create_ta, clean_ta

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

# ENVIRONMENT
from lib.envs.simulated import SimulatedEnv
from lib.envs.live import LiveEnv

# UTILS
from lib.utils.added_tools import maybe_make_dir, dir_setup, generate_actions

# GLOB VARIABLES
ACTIONS = ["SELL", "HOLD", "BUY"]
MINI_DATA_PATH = "./data/ta_all_mini_clean.csv"
TOTAL_DATA_PATH = "./data/ta_all_clean.csv"

def manual_agent_params():
    return {
        'n_steps': 1024,
        'gamma': 0.9391973108460121,
        'learning_rate': 0.00010179263199758284,
        'ent_coef': 0.0001123894292050861,
        'cliprange': 0.2668120684510983,
        'noptepochs': 5,
        'lam': 0.8789545362092943
    }

def historical_yahoo(stock):
  data = si.get_data(stock, end_date = pd.Timestamp.today() + pd.DateOffset(10))
  data = create_ta(data)
  data = data.fillna(0)
  data = clean_ta(data)
  return data

def train_val_test_split(path):
  data = pd.read_csv(path)
  n_features = data.shape[1]
  train_data = data.iloc[:-50000, :]
  validation_data = data.iloc[-50000:-5000, :]
  test_data = data.iloc[-5000:, :]
  return train_data, validation_data, test_data

def train_val_test_split_finetune(data):
  n_features = data.shape[1]
  train_data = data.iloc[:-2000, :]
  test_data = data.iloc[-2000:, :]
  return train_data, test_data

def make_env(env, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

class Static_Session:
  def __init__(self, mode, test_episodes, initial_invest, session_name, stock=None, brain=None):
    # SESSION_VARIABLES
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
    self.brain = brain
    self.n_steps_per_eval = 5000000
    self.logger = init_logger(__name__, show_debug=True)
    self.stock = stock
    coloredlogs.install(level='TEST')
    
    self.train_data, self.validation_data, self.test_data = train_val_test_split(TOTAL_DATA_PATH)
    
    self.optuna_study = optuna.create_study(
           study_name="{}_study".format(self.session_name), 
           storage="sqlite:///data/params.db", 
           load_if_exists=True,
           pruner=optuna.pruners.MedianPruner())
    
    self.logger.debug('Initialized Static Session: {}'.format(self.session_name))
    self.logger.debug('Mode: {}'.format(self.mode))

  # OUTPUT FUNCTIONS   
  def print_stats(self, e, info):
    # Print stats
    self.logger.info("episode: {}/{}, episode end value: {}".format(
      e + 1, self.test_episodes, info[0]['cur_val']))
    self.logger.info("loss percent --> {}%".format(int(self.losses / (e + 1) * 100)))
  def write_to_story(self,f, action, info):
    f.write("{},{},{},{},{},{}\n".format(ACTIONS[self.actions[action][0]],
                                              self.actions[action][1], 
                                              info['owned_stocks'], 
                                              info['cash_in_hand'], 
                                              info['cur_val'], 
                                              info['price']))
  # OPTIMIZE FUNCTIONS
  def optimize_agent_params(self, trial):
    return {
        'n_steps': int(trial.suggest_loguniform('n_steps', 512, 2048)),
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
                verbose=0, 
                tensorboard_log="./logs/", 
                nminibatches=1,
                **model_params)
    # TODO fix this
    for eval_idx in range(n_prune_evals_per_trial):
        try:
            model.learn(self.n_steps_per_eval)
        except AssertionError:
            raise
 
        last_reward = self.run_test(model)

        trial.report(-1 * last_reward, eval_idx)
        
        if trial.should_prune(eval_idx):
            raise optuna.structs.TrialPruned()
    return -1 * last_reward
  def get_model_params(self):
    params = self.optuna_study.best_trial.params
    self.logger.debug('Loaded best parameters as: {}'.format(params))

    return {
        'n_steps': int(params['n_steps']),
        'gamma': params['gamma'],
        'learning_rate': params['learning_rate'],
        'ent_coef': params['ent_coef'],
        'cliprange': params['cliprange'],
        'noptepochs': int(params['noptepochs']),
        'lam': params['lam'],
    } 
  def run_optimization(self, n_trials: int = 10):
    try:
      self.optuna_study.optimize(self.optimize_params, n_trials=n_trials, n_jobs=1)
    except KeyboardInterrupt:
      pass
    self.logger.info('Finished trials: {}'.format(len(self.optuna_study.trials)))
    self.logger.info('Best trial: {}'.format(self.optuna_study.best_trial.value))
    return self.optuna_study.trials_dataframe()
  # TRAINING AND TESTING
  def run_test(self,model, validation = True, finetune=False, out_file=False, verbose=True):  
    
    f = None
    if out_file:
      f = open("stories/{}-{}-{}.csv".format(self.timestamp, self.mode, "BTC"),"w+")
      f.write("OPERATION,AMOUNT,STOCKS_OWNED,CASH_IN_HAND,PORTFOLIO_VALUE,OPEN_PRICE\n")
    
    env = None

    if not finetune:
      if validation: 
        env = DummyVecEnv([lambda: SimulatedEnv(self.validation_data, self.initial_invest, self.mode)])
      else:
        env = DummyVecEnv([lambda: SimulatedEnv(self.test_data, self.initial_invest, self.mode)])
    else:
        data = historical_yahoo("NKE")
        self.logger.debug('Downloaded Data from yahoo finance.')

        _ , test_data = train_val_test_split_finetune(data)
        env = DummyVecEnv([lambda: SimulatedEnv(test_data, self.initial_invest, self.mode)])
        self.logger.debug('Downloaded Data from yahoo finance.')

    total_reward = []

    for e in range(self.test_episodes):
      # Reset the environment at every episode. 
      state = env.reset()      
      # Initialize variable to get reward stats. 
      
      for time in range(0, 180):
        action, _states = model.predict(state)
        next_state, reward, done, info = env.step(action)
        
        if out_file:
          self.write_to_story(f, action[0], info[0])

        total_reward.append(reward)
        state = next_state
        
        if done:
          if info[0]['cur_val'] < self.initial_invest:
            self.losses = self.losses + 1
          if verbose:
            self.print_stats(e, info)
          if out_file:
            f.write("{},{},{},{},{},{}\n".format(-1,-1,-1,-1,-1,-1))
          break
    self.losses = 0
    return np.mean(total_reward) 
  def run_train(self):
    train_env = DummyVecEnv([lambda: SimulatedEnv(self.train_data, self.initial_invest, self.mode)])

    #model_params = self.get_model_params()
    model_params = manual_agent_params()
    model = PPO2(MlpPolicy, 
                train_env, 
                verbose=1, 
                nminibatches=1,
                tensorboard_log="./logs/", 
                **model_params)
    try:
      model.learn(total_timesteps=self.n_steps_per_eval)
      result = self.run_test(model, validation=False)
      
      self.logger.info("test_mean --> {}".format(result))
      
      model.save("{}.pkl".format(self.session_name))

    except KeyboardInterrupt:
      print("Saving model...")
      model.save("{}.pkl".format(self.session_name))
  def fine_tune(self, stock, model_path, ts=1000000):
      assert self.brain is not None
      
      data = historical_yahoo(stock)
      train_data, _ = train_val_test_split_finetune(data)
      
      fine_tune_env = SimulatedEnv(train_data, self.initial_invest, self.mode)
      fine_tune_env = DummyVecEnv([lambda: fine_tune_env])

      model = PPO2.load(model_path)
      model.set_env(fine_tune_env)
    
      self.logger.info("Finetuning for {}...".format(stock))
      
      model.learn(total_timesteps=ts)
      model.save("{}__{}.pkl".format(self.session_name, stock))
      self.logger.info("Saved as {}__{}.pkl".format(self.session_name, stock))

      result = self.run_test(model, finetune=True)

