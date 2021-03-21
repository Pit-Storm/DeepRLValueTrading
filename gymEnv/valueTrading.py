from gym import Env
from gym import spaces
from gym.utils import seeding
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime
import json
from pathlib import Path, PurePath
import config

class valueTradingEnv(Env):
    """
    Inherits from gym.Env. It is a stock trading Environment

    params:
        - df (pandas DataFrame): DF with 'date' and 'symbol' set as multilevel index and at least opening-, closing-prices and 0/1 column if stock is tradeable (1 is tradable) as first, second and third column. IMPORTANT: Open, Close and tradeable always has to be as first, second and third columns respectively for taking actions and calculating reward.
        - sample (bool): Should the env sample the trading period (episode) out of df in length of yearrange.
        - yearrange (int): How many years will one episode take? DF has to be at least this range.
        - cagr (boo): Should reward be calculated by CAGR=
        - episodic (bool): Will the Agent get sparse rewards (only on end of episode).
        ...
    returns:
        A gym.Env object.
    """
    metadata = {'render.modes': "human"}
    
    def __init__(self, df: pd.DataFrame, sample: bool=False, episodic: bool=False,
                save_path: Path=None, yearrange: int=4, cagr: bool=False, seeding: bool=None):
        # self variables
        self.df = df
        self.sample = sample
        self.seeding = seeding
        self.episodic = episodic
        self.save_path = save_path
        self.yearrange = yearrange
        self.cagr = cagr
        self.df_dt_filter = self.df.index.get_level_values(level="date")
        self.indicators = self.df.columns.tolist()
        self.init_time =  datetime.now().strftime('%H-%M-%S-%f')
        self.num_symbols = len(self.df.index.get_level_values(level="symbol").unique().tolist())
        self.num_eps = 0
        self.fee = config.TRADE_FEE_PRCT
        self.scaling = config.ACTION_SCALING
        self.init_cash = config.INIT_CASH

        # Vars not yet set
        self.cost = None
        self.data = None
        self.data_dt_filter = None
        self.data_dt_unique = None
        self.date = None
        self.date_idx = None
        self.done = None
        self.end_date = None
        self.episode_totalValues = None
        self.info = None
        self.new_state = None
        self.state = None

        # If we want to sample during training, we have to seed the RNG
        if self.sample:
            assert not isinstance(self.seeding, type(None)), "If you want to sample, you have to set a seeding."
            self.seed(seed=self.seeding)

        # Spaces
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.num_symbols,))
        obs_shape = 1 + self.num_symbols + (len(self.indicators) * self.num_symbols)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape = (obs_shape,))

        # Create env_info save path
        if not isinstance(self.save_path, type(None)):
            assert isinstance(self.save_path, PurePath), "save_path is no pathlib.Path object."
            self.save_path.mkdir(parents=True, exist_ok=True)

    def _sell_stock(self, num, index):
        # Are we selling less or equal num of stocks we have?
        if self.new_state[1+index] >= num:
            # get open price of stock to calculate amount
            price = self.new_state[1+self.num_symbols+index]
            amount = price * num
            # calculate cost
            self.cost[index] = amount * self.fee
            # put the stock from portfolio
            self.new_state[1+index] -= num
            # recalculate the cash
            self.new_state[0] += (amount - self.cost[index])
        else:
            pass
    
    def _buy_stock(self, num, index):
        # get open price of stock
        price = self.new_state[1+self.num_symbols+index]
        amount = price * num
        # calculate cost
        self.cost[index] = amount * self.fee

        # Check if we have enough cash
        if self.new_state[0] >= amount+self.cost[index]:
            # call the stock into portfolio
            self.new_state[1+index] += num
            # update the cash
            self.new_state[0] -= (amount + self.cost[index])
        else:
            pass

    def _get_time_range(self):
        # get all unique dates in df
        dates = self.df_dt_filter.unique()
        if self.sample:
            # set max end date to 4 years befor max date
            sample_end = dates.max() - relativedelta(years=self.yearrange)
            sample_begin = dates.min()
            # throw away all dates out of begin and end
            dates = dates[dates.slice_indexer(sample_begin, sample_end)].tolist()
            # sample start date randomly out of possible dates
            start_date = self.np_random.choice(dates)
            # set end date num-yrs minus 1day relative to start date
            end_date = start_date + relativedelta(years=self.yearrange,days=-1)
        else: # If we are not in train environment
            # Set start date and end date to min and max of df respectively
            start_date = dates.min()
            end_date = dates.max()
        
        return (start_date, end_date)

    def step(self, action):
        """
        Run one timestep of the environment's dynamics.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.cost = [0]*self.num_symbols
        # Skale action by item
        action = np.array([int(item*self.scaling) for item in action])
        # real_action = action.copy() # save for info

        # count date one day ahead
        self.date_idx += 1
        self.date = self.data_dt_unique[self.date_idx]
        # Check done conditions
        # Is date equal to end_date?
        if self.date == self.end_date:
            self.done = True

        # set up new state based on current state
        # We manipulate the portfolios cash and number of shares in new_state when buying and selling
        # we need the old portfolio balance to calculate the reward
        self.new_state =    [self.state[0]] + \
                            self.state[1:(1+self.num_symbols)] + \
                            [item for indicator in self.indicators for item in self.data[self.data_dt_filter == self.date][indicator].values.tolist()]
        
        # When tradeable is 0 Set action of stock  to 0
        # Because it is not tradable at this state
        for idx in range(len(action)):
            tradeable = self.new_state[1+self.num_symbols*3+idx]
            if tradeable == 0:
                action[idx] = 0
        # return sorted actions indices in sorted order (lowest first)
        argsort_actions = np.argsort(action)
        # get indices of sell actions
        sell_indices = argsort_actions[:np.where(action < 0)[0].shape[0]]
        # get indices of buy actions
        buy_indices = argsort_actions[::-1][:np.where(action > 0)[0].shape[0]]

        # perform each sell action
        for idx in sell_indices:
            self._sell_stock(int(action[idx]*-1), idx)
        # perform each buy action
        for idx in buy_indices:
            self._buy_stock(int(action[idx]), idx)

        # Is the cash lower than x% of init_cash?
        # If we set this to 0.0 we allow to don't hold cash anyway
        if self.new_state[0] < self.init_cash*0:
            self.done = True

        ### calculate reward
        # We calculate the total value from the closing price
        self.episode_totalValues[self.date_idx] = self.new_state[0] + \
                sum(np.array(self.new_state[1:(1+self.num_symbols)]) * \
                    np.array(self.new_state[(1+self.num_symbols*2):(1+self.num_symbols*3)]))
        # We use growth rates to avoid volatility
        if self.episodic and not self.done:
            step_reward = 0.0
        else:
            if self.cagr:
                # Compound growth rate towards actual step
                step_reward = ((self.episode_totalValues[self.date_idx] / self.episode_totalValues[0]) ** (1/self.date_idx)) - 1.0
            else:
                # Growth rate / Rate of return towards actual step
                step_reward = self.episode_totalValues[self.date_idx] / self.episode_totalValues[0] - 1.0

        # set new_state as current state
        self.state = self.new_state

        # add the values to the info container
        self.info["dates"].append(self.date.strftime("%Y-%m-%d"))
        self.info["cashes"].append(self.state[0])
        self.info["numShares"].append(self.state[1:(1+self.num_symbols)])
        self.info["totalValues"].append(self.episode_totalValues[self.date_idx])
        # self.info["realActions"].append(real_action.tolist())
        self.info["rewards"].append(step_reward)
        # self.info["costs"].append(self.cost)

        # Strip out the actual and the t-1 info to return it
        step_info = {key: value[-2:] for key,value in self.info.items()}

        if self.done:
            # This is because the agent would not do another step if the episode is done
            # So we need to append some zeros to make the lists the identical lengths
            # self.info["realActions"].append([0]*self.num_symbols)
            self.info["rewards"].append(0)
            # self.info["costs"].append(0)

            # Count a episode
            self.num_eps += 1
            # Save info container to json in path if one is given
            if not isinstance(self.save_path, type(None)):
                filename = str(self.init_time) + "_episode-" + str(self.num_eps).rjust(4, "0") + ".json"
                jsonpath = self.save_path.joinpath(filename)
                with open(jsonpath, 'w') as fp:
                    json.dump(self.info, fp)

        return (self.state, step_reward, self.done, step_info)

    def reset(self):
        """Resets the environment to an initial state and returns an initial
        observation.

        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.

        Returns:
            observation (object): the initial observation.
        """
        # Get date range
        start_date, end_date = self._get_time_range()
        # slice episode data out of df
        self.data = self.df[(self.df_dt_filter >= start_date) & (self.df_dt_filter <= end_date)]
        # get a filter object out of index level date
        self.data_dt_filter = self.data.index.get_level_values(level="date")
        self.data_dt_unique =  self.data_dt_filter.unique().tolist()
        # Reset date_idx
        self.date_idx = 0
        # get first date object
        self.date = self.data_dt_unique[self.date_idx]
        # set real end date
        self.end_date = self.data_dt_unique[-1]
        self.done = False
        self.episode_totalValues =pd.Series(np.zeros(shape=len(self.data_dt_unique)))

        # generate first state
        self.state =    [self.init_cash] + \
                        [0]*self.num_symbols + \
                        [item for indicator in self.indicators for item in self.data[self.data_dt_filter == self.date][indicator].values.tolist()]
        
        self.episode_totalValues[0] = self.state[0]

        # info container for rendering and output
        self.info = {
            "dates": [self.date.strftime("%Y-%m-%d")],
            "cashes": [self.state[0]],
            "numShares": [self.state[1:(1+self.num_symbols)]],
            "totalValues": [self.episode_totalValues[self.date_idx]],
            # "realActions": [],
            "rewards": []
            # "costs": []
        }

        return self.state
    
    def render(self, mode='human'):
        """
        Prints or plots some basic information.

        Args:
            mode (str): the mode to render with
        """
        if mode == 'human':
            print("Here should be a basic plot...")
        else:
            super(valueTradingEnv, self).render(mode=mode) # just raise an exception

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
