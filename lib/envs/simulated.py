import gym
import numpy as np
import random
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib 

from lib.utils.added_tools import generate_actions, clamp

class SimulatedEnv(gym.Env):
  
  metadata = {'render.modes': ['human', 'system', 'none']}

  def __init__(self, train_data, init_invest=1000, mode="test"):

    super(SimulatedEnv, self).__init__()
    
    self.dataset = train_data
    self.mode = mode
    self.n_steps = 500
    self.finish_step = 0
    self.init_invest = init_invest
    self.cur_step = None
    self.current_state = []
    self.owned_stocks = None
    self.cash_in_hand = None    

    self.scaler = MinMaxScaler()
    self.scaler.fit(self.dataset)
    
    if mode is "train":
      joblib.dump(self.scaler, 'train_scaler.pkl') 
    
    if mode is "test" or mode is "validation":
      self.scaler = joblib.load('train_scaler.pkl') 

    self.actions = generate_actions()
    self.action_space = gym.spaces.Discrete(30)  

    self.n_features = train_data.shape[1]
    self.obs_shape = (1, self.n_features)
    self.observation_space = gym.spaces.Box(low=0, 
                                            high=1,
                                            shape=self.obs_shape, 
                                            dtype=np.float16)  
    self.reset()

  #TODO implement seeding. 
  # def _seed(self, seed=None):
  #   self.np_random, seed = seeding.np_random(seed)
  #   return [seed]

  def reset(self):
    limit = len(self.dataset) - self.n_steps
    self.cur_step = random.randrange(limit)
    self.finish_step = self.cur_step + self.n_steps
    
    self.owned_stocks = 0
    self.cash_in_hand = self.init_invest
    self.current_state = self.dataset.iloc[self.cur_step,:]

    return self._get_obs()

  def step(self, action):
    assert self.action_space.contains(action)
 
    prev_val = self._get_val()
    
    self.cur_step += 1
    self.current_state = self.dataset.iloc[self.cur_step,:] 
    open_price = self.current_state[0]

    # Trade with an action 
    self._trade(action)
    
    # Check on how much it changed
    cur_val = self._get_val()

    reward = clamp(-5, cur_val - prev_val, 5)
    #reward = cur_val - prev_val

    done = (self.cur_step >= self.finish_step - 1) 
    info = {'cur_val': cur_val, 
            'prev_val': prev_val, 
            'action': action, 
            'owned_stocks': self.owned_stocks, 
            'cash_in_hand': self.cash_in_hand, 
            'price': open_price }

    return self._get_obs(), reward, done, info

  def _get_obs(self):
    obs = []
    for element in self.current_state:
      obs.append(element)
    
    obs = np.array(obs)
    return self.scaler.transform([obs])
  def _get_val(self):
    open_price = self.current_state[0]
    return np.sum(self.owned_stocks * open_price) + self.cash_in_hand

  def _trade(self, action):
    combo = self.actions[action]
    move = combo[0]
    amount = combo[1]
    
    open_price = self.current_state[0]
    expense = amount * open_price
    
    sell = False
    buy = False
    
    if move == 0:
      sell = True
    elif move == 2:
      buy = True
    
    if sell and self.owned_stocks >= amount:
      self.cash_in_hand += expense
      self.owned_stocks -= amount
    
    if buy and self.cash_in_hand >= expense:
      self.cash_in_hand -= expense
      self.owned_stocks += amount 
  
  def render(self, mode='system'):
    if mode == 'system':
        print('Price: ' + str(self.current_state[0]))
        print('Net worth: ' + str(self._get_val))

  def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None