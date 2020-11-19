from gym import Env
from gym import spaces
import numpy as np
import pandas as pd
from .. import config

class valueTradingEnv(Env):
    """
    Inherits from gym.Env. It is a stock trading Environment

    params:
        - df: pandas DataFrame with 'date' and 'symbol' set as multilevel index and at least adjusted closing price as column. IMPORTANT: Adjusted Closing price always has to be the first column for calculating portfolio amount
    
    returns:
        A gym.Env object.
    """
    metadata = {'render.modes': "human"}
    spec = None
    
    def __init__(self, df):
        # self variables
        self.df = df
        self.indicators = self.df.columns.tolist()
        self.num_symbols = len(self.df.index.get_level_values(level="symbol").unique().tolist())
        self.dt_filter = self.df.index.get_level_values(level="date")
        self.fee = config.TRADE_FEE_PRCT
        self.scaling = config.ACTION_SCALING

        # Spaces
        self.action_space = spaces.Box(low = -1, high = 1,shape = (self.num_symbols,))
        obs_shape = 1 + self.num_symbols + (len(self.indicators) * self.num_symbols)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape = (obs_shape,))

    def _sell_stock(self, action):
        raise NotImplementedError

    def _buy_stock(self, action):
        raise NotImplementedError

    def _calc_reward(self):
        raise NotImplementedError

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.reard = 0
        self.cost = 0

        # TODO: count date one day ahead
        self.date = 

        # set done value
        self.done = self.date == self.end_date


        return self.state, self.reward, self.done, self.info

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
        self.date = 
        self.end_date = 
        self.data = self.df[(self.dt_filter >= self.date) & (self.dt_filter <= self.end_date)]
        self.done = False
        
        self.state =    [config.INIT_CASH] + \
                        [0]*self.num_symbols + \
                        [item for indicator in self.indicators for item in self.data[self.dt_filter == self.date][indicator].values.tolist()]
        
        # info container
        self.info = {
            "dates": [self.date],
            "actions": [],
            "rewards": [],
            "balance": [self.state[0]],
            "numshares": [self.state[1:1+self.num_symbols]],
            "prices": [self.state[1+self.num_symbols*2:1+self.num_symbols*3]]
            "costs": []
        }

        return self.state
    
    def render(self, mode='human'):
        """Renders the environment.

        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.) By convention,
        if mode is:

        - human: render to the current display or terminal and
          return nothing. Usually for human consumption.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3),
          representing RGB values for an x-by-y pixel image, suitable
          for turning into a video.
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation. The text can include newlines
          and ANSI escape sequences (e.g. for colors).

        Note:
            Make sure that your class's metadata 'render.modes' key includes
              the list of supported modes. It's recommended to call super()
              in implementations to use the functionality of this method.

        Args:
            mode (str): the mode to render with

        Example:

        class MyEnv(Env):
            metadata = {'render.modes': ['human', 'rgb_array']}

            def render(self, mode='human'):
                if mode == 'rgb_array':
                    return np.array(...) # return RGB frame suitable for video
                elif mode == 'human':
                    ... # pop up a window and render
                else:
                    super(MyEnv, self).render(mode=mode) # just raise an exception
        """
        raise NotImplementedError

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.

        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        pass

    def seed(self, seed=None):
        """Sets the seed for this env's random number generator(s).

        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.

        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        return
