import numpy as np
import pandas as pd
from gymnasium import spaces, Env

# SimExBase is the parent, general class for the rest of the SimEx classes
class SimExBase(Env):
    def __init__(self, 
                 data_path, 
                 memory_size=72, 
                 episode_length=720, 
                 val_split=0.10, 
                 val_dist=4, 
                 gamma=0, 
                 seed=None, 
                 verbose=1):
        
        # we call the superconstructor from the gym.Env class
        super(SimExBase, self).__init__()

        # we set the observation space
        self.memory_size = memory_size  
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.memory_size,), dtype=np.float32)

        # we load the historical data
        self.data = pd.read_parquet(data_path)
        self.data_size = self.data.shape[0]
        
        # we set the episode length and validation episodes
        self.episode_length = episode_length
        n_val = np.floor(val_split * self.data_size / self.episode_length, dtype=np.float32).astype(np.int8)
        self.val_indices = self._gen_val_indices(n_val, val_dist)
        
        # we set the updating parameter for the reference balance
        self.gamma = gamma

        # we call the reset method to initiate an episode
        self.reset(seed)    
        self.verbose = verbose



    ################################################## Public methods ##################################################
    # step() performs an updating step of the environment, given an action
    def step(self, action):
        # we update the ep_counter and current_idx parameters
        self.ep_counter += 1
        self.current_idx += 1

        # we save the previous total_balance and update the balances using _update_balance()
        prev_balance = self.total_balance
        self._update_balance(action)
        
        # we get the to be returned observation
        observation = self._get_observation()
        
        # we compute the reward
        reward = (self.total_balance - prev_balance) / self.ref_balance
        self.ep_reward += reward

        if self.verbose==1 and self.ep_counter==self.episode_length:
            print("Final Episode Balance = %f - Episode Reward = %f - All-in Reward = %f" 
                  % (self.total_balance, self.ep_reward, self.ep_avg_rate * self.episode_length))

        return observation, reward, self.ep_counter==self.episode_length, False, {}
    

    # reset() resets the environment to the beginning of a new episode
    def reset(self, seed=None, starting_idx=None):
        # we set the seed for the RNG
        np.random.seed(seed=seed)

        # we get the starting index for the episode
        if starting_idx==None:
            self.starting_idx = self._get_starting_idx()
        else:
            self.starting_idx = starting_idx

        # we reset the episode parameters
        self.ep_counter = 0
        self.ep_reward = 0
        self.current_idx = self.starting_idx

        # we reset the balance and recalculate the episode metrics
        self._reset_balance() 
        self._get_episode_rate()

        # we get the to be returned observation
        observation = self._get_observation()
        
        return observation, {}


    # render() graphycally renders the environment
    def render(self, mode='human'):
        pass # TO DO: code this later

    # close() closes the environment
    def close(self):
        pass # TO DO: code this later





    ################################################## Private methods #################################################
    # _update_balance() updates the balance of the environment according to a performed action
    def _update_balance(self, action):
        # we get the liquidity
        liquidity = self.total_balance - self.asset_balance

        # we get the balance and liquidity change performed by the action
        change = action * self.total_balance
        change = max(-self.asset_balance, min(change, liquidity))

        # we update the liquidity and asset balance
        self.asset_balance += change
        liquidity -= change

        # we get the asset prices before and after the action
        prev_observation = self.data.iloc[self.current_idx-1].to_dict()
        next_observation = self.data.iloc[self.current_idx].to_dict()

        # we update the asset balance with the new prices
        self.asset_balance += self.asset_balance * \
                                (next_observation['Close'] - prev_observation['Close']) / prev_observation['Close']
        
        # we update the total and reference balance (using a moving average)
        self.total_balance = self.asset_balance + liquidity
        self.ref_balance = self.gamma * self.total_balance + (1-self.gamma) * self.ref_balance



    # _reset_balance() resets the balance of the enviorment assigning a random amount to the asset
    def _reset_balance(self):
        # we reset the total and reference balances to one and randomize the ratio placed in the asset
        self.total_balance = 1
        self.asset_balance = np.random.random()
        self.ref_balance = 1


    # _get_observation() returns the observation of the current index
    def _get_observation(self):
        observation = np.array(self.data.iloc[(self.current_idx-self.memory_size):self.current_idx]['Close'], 
                                dtype=np.float32)
        return observation
    

    def _get_episode_rate(self):
        self.ep_avg_rate = \
            (self.data.iloc[self.current_idx+self.episode_length]['Close'] - self.data.iloc[self.current_idx]['Close'])\
            / (self.data.iloc[self.current_idx]['Close'] * self.episode_length)
        

    # _get_starting_idx() generates a starting index for training
    def _get_starting_idx(self):
        # we calculate possible starting indices until we get to one that is valid
        not_valid = True
        while not_valid:
            # we generate a random starting index and check if it overlaps with any validation episode
            starting_idx = np.random.randint(self.memory_size, self.data_size - self.episode_length)
            val_overlap = (np.abs((starting_idx - self.val_indices)) < self.episode_length).any()
            if not(val_overlap):
                not_valid = False
        
        return starting_idx


    # _gen_val_indices() generates the validation indices for the environment
    def _gen_val_indices(self, n_val, val_dist):
        # we calculate possible validation indices until we get to a set of them that is valid
        not_valid = True
        while not_valid:
            # we generate random validation indices and check that their episodes do not overlap
            val_indices = np.sort(np.random.randint(self.memory_size, self.data_size - self.episode_length, size=n_val))
            if not (np.diff(val_indices) < (val_dist * self.episode_length)).any():
                not_valid = False

        return val_indices





# SimExDiscrete is the class for discrete action spaces
class SimExDiscrete(SimExBase):
    def __init__(self, action_values=[-0.10, 0, 0.10], memory_size=1440, episode_length=10080, val_split=0.10, 
                    val_dist=4, gamma=0.01, seed=None, verbose=1, data_path='./data_raw/parquet/XBTUSDT_1.parquet'):
        super(SimExDiscrete, self).__init__(memory_size=memory_size,
                                            episode_length=episode_length, 
                                            val_split=val_split, 
                                            val_dist=val_dist, 
                                            gamma=gamma, 
                                            seed=seed, 
                                            verbose=verbose,
                                            data_path=data_path)
        
        # we set the action space        
        self.action_space = spaces.Discrete(len(action_values))
        self.action_values = action_values
    

    def _update_balance(self, action):
        super()._update_balance(self.action_values[action])



# SimExContinuous is the class for discrete action spaces
class SimExContinuous(SimExBase):
    def __init__(self, memory_size=1440, episode_length=10080, val_split=0.10, val_dist=4, gamma=0.01, seed=None, 
                 verbose=1, data_path='./data_raw/parquet/XBTUSDT_1.parquet'):
        super(SimExContinuous, self).__init__(memory_size=memory_size, 
                                            episode_length=episode_length, 
                                            val_split=val_split, 
                                            val_dist=val_dist, 
                                            gamma=gamma, 
                                            seed=seed, 
                                            verbose=verbose,
                                            data_path=data_path)
        
        # we set the action space        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
    

    def _update_balance(self, action):
        super()._update_balance(action.item())