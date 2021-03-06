import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import numpy as np
import logging
import os

from data import handling as dth
from gymEnv.valueTrading import valueTradingEnv
import config
from algos.basic import buyHold

from stable_baselines import A2C, PPO2, DDPG
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.evaluation import evaluate_policy

def DRL() -> None:
    ### PREPARATION
    # callback for validation
    eval_callback = EvalCallback(val_env, best_model_save_path=config.val_path,
                             log_path=config.val_path, eval_freq=config.val_freq, verbose=config.verbosity,
                             deterministic=config.deterministic, n_eval_episodes=config.val_eps)

    ### SETUP AND TRAIN
    # Setup model
    if config.MODEL_NAME == "A2C":
        model = A2C(config.POLICY, train_env, verbose=config.verbosity, tensorboard_log=config.tb_path, seed=config.seed)
    elif config.MODEL_NAME == "PPO":
        mbatches = config.num_envs // 2 if config.num_envs % 2 == 0 else 1
        model = PPO2(config.POLICY, train_env, verbose=config.verbosity, tensorboard_log=config.tb_path, nminibatches=mbatches, seed=config.seed)
    elif config.MODEL_NAME == "DDPG":
        # the noise objects for DDPG
        n_actions = train_env.action_space.shape[-1]
        param_noise = None
        # action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=float(0.4) * np.ones(n_actions))
        model = DDPG(config.POLICY, train_env, param_noise=param_noise, action_noise=action_noise, verbose=config.verbosity, tensorboard_log=config.tb_path, seed=config.seed)

    logger.warn(f"{os.getpid()} | {config.MODEL_NAME} Model created. Starting to learn...")
    ###
    # Train Model
    model = model.learn(total_timesteps=config.learn_steps, callback=eval_callback)

    logger.warn(f"{os.getpid()} | Training of {config.MODEL_NAME} done. Loading best model for test...")
    # Load best model after training
    if config.MODEL_NAME == "A2C":
        model = A2C.load(load_path=config.val_path.joinpath("best_model.zip"))
    elif config.MODEL_NAME == "PPO":
        model = PPO2.load(load_path=config.val_path.joinpath("best_model.zip"))
    elif config.MODEL_NAME == "DDPG":
        model = DDPG.load(load_path=config.val_path.joinpath("best_model.zip"))

    logger.info(f"Best model loaded. Start testing...")
    ### EVAL MODEL
    # Make prediction in test_env
    _ = evaluate_policy(model=model, env=test_env, deterministic=config.deterministic,
                                n_eval_episodes=config.test_eps, return_episode_rewards=True)

    logger.warn(f"{os.getpid()} | Testing done.")

def basic() -> None:
    ###
    # Setup ENV
    env = test_env

    ep_rewards = []

    logger.warn(f"{os.getpid()} | Start trading with {config.MODEL_NAME} algo.")
    for episode in range(config.test_eps):
        state = env.reset() # reset for each new episode
        done = False
        while not done: # run until done
            if config.MODEL_NAME == "RANDOM":
                action = env.action_space.sample() # select a random action
            if config.MODEL_NAME == "BUYHOLD":
                action = buyHold(state[0], env.action_space)
            state, reward, done, _ = env.step([action])
        ep_rewards.append(reward[0])
        if config.verbosity: logger.info(f"{episode+1}. Episode last reward: {ep_rewards[-1]}")
        if (episode+1) % 10 == 0 or (episode+1) == config.test_eps:
            mean_reward = sum(ep_rewards) / len(ep_rewards)
            logger.info(f"Mean reward after {episode+1}th Episode: {mean_reward}")

    logger.warn(f"{os.getpid()} | Trading with {config.MODEL_NAME} algo done.")

if __name__ == "__main__":
    # Create logger
    logger = logging.getLogger(__name__)
    logger.propagate = 0
    logger.addHandler(config.log_handler_file)
    logger.addHandler(config.log_handler_std)
    logger.warn(f"{os.getpid()} | Script started.")

    ### DATA
    # Load Dataset
    stocks_df = dth.load_data(config.data_path)

    # make train, val, test df
    train, val, test = dth.train_val_test_split(df=stocks_df, years=config.yearrange)

    logger.info(f"Data loaded and split.")

    # Training Env
    train_env = DummyVecEnv([(lambda: valueTradingEnv(df=train, sample=config.trainsampling, episodic=config.episodic, yearrange=config.yearrange,
                            cagr=config.cagr, save_path=config.env_path.joinpath("train"))) for i in range(config.num_envs)])

    # Validation Env
    val_env = DummyVecEnv([lambda: valueTradingEnv(df=val, sample=False, episodic=config.episodic, yearrange=config.yearrange,
                            cagr=config.cagr)])

    # test_env
    test_env = DummyVecEnv([lambda: valueTradingEnv(df=test, sample=False, episodic=config.episodic, yearrange=config.yearrange,
                            cagr=config.cagr, save_path=config.env_path.joinpath("test"))])

    logger.info(f"Environments created.")

    # Call the specific Model function
    if config.MODEL_NAME in config.drl_algos:
        # Setup stacked envs if enabled
        if config.NUM_STACKS > 0:
            train_env = VecFrameStack(train_env, n_stack=config.NUM_STACKS)
            val_env = VecFrameStack(val_env, n_stack=config.NUM_STACKS)
            test_env = VecFrameStack(test_env, n_stack=config.NUM_STACKS)
        DRL()
    elif config.MODEL_NAME in config.basic_algos:
        basic()
    
    logger.warn(f"{os.getpid()} | Script done.")