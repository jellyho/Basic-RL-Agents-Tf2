import numpy as np
import pandas as pd
import random
from collections import defaultdict
import matplotlib.pyplot as plt

class QLearning:
  def __init__(self, actions):
    self.ACTIONS = np.array(actions)
    self.step_size = 0.01
    self.discount_factor = 0.9
    self.epsilon = 0.1
    self.q_table = defaultdict(lambda:[0.0 for _ in range(len(actions))])

  def get_action(self, state):
    if np.random.rand() < self.epsilon:
        action = np.random.choice(self.ACTIONS)
    else:
        q_list = self.q_table[str(state)]
        action = self.ACTIONS[np.random.choice(np.argwhere(q_list == np.amax(q_list)).flatten().tolist())]
    return action

  def save(self, root):
    df = pd.DataFrame([[state, self.q_table[state]] for state in self.q_table.keys()], columns=['states', 'q_list'])
    df.to_csv(root)

  def load(self, root):
    def str_to_list(s):
      s = s.split('[')
      s = s[1].split(']')
      s = s[0].split(', ')
      for i in range(len(s)):
        s[i] = float(s[i])
      return s
    
    self.q_table = defaultdict(lambda:[0.0 for _ in range(len(self.Actions))])
    df = pd.read_csv(root)
    for idx in df.index:
      self.q_table[df['states'][idx]] = str_to_list(df['q_list'][idx])
      
  def learn(self, state, action, reward, next_state):
    
    state, next_state = str(state), str(next_state)
    
    current_q = self.q_table[state][np.argwhere(self.ACTIONS==action)[0][0]]
    next_state_q = max(self.q_table[next_state])

    td = reward + self.discount_factor * next_state_q - current_q
    new_q = current_q + self.step_size * td
    self.q_table[state][np.argwhere(self.ACTIONS==action)[0][0]] = new_q
