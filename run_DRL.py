from data import handling as dth
from gymEnv.valueTrading import valueTradingEnv
import config
from algos.basic import buyHold

from stable_baselines import A2C, PPO2, DDPG
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.evaluation import evaluate_policy

from os import system
import numpy as np

system("clear")

### DATA
# Load Dataset
stocks_df = dth.load_data(config.data_path)

# make train, val, test df
train, val, test = dth.train_val_test_split(stocks_df)

# Training Env
train_env = DummyVecEnv([lambda: valueTradingEnv(df=train, sample=config.trainsampling, episodic=config.episodic, yearrange=config.yearrange,
                        save_path=config.env_path.joinpath("train")) for i in range(config.num_envs)])
train_env = VecCheckNan(train_env, raise_exception=True)

# Validation Env
val_env = DummyVecEnv([lambda: valueTradingEnv(df=val, sample=False, episodic=config.episodic, yearrange=config.yearrange,
                        save_path=config.env_path.joinpath("val")) for i in range(config.num_envs)])
val_env = VecCheckNan(val_env, raise_exception=True)

# test_env
test_env = DummyVecEnv([lambda: valueTradingEnv(df=test, sample=False, episodic=config.episodic, yearrange=config.yearrange,
                        save_path=config.env_path.joinpath("test"))])
test_env = VecCheckNan(test_env, raise_exception=True)


def DRL() -> None:
    ### PREPARATION
    # callback for validation
    eval_callback = EvalCallback(val_env, best_model_save_path=config.val_path,
                             log_path=config.val_path, eval_freq=config.val_freq,
                             deterministic=config.deterministic, n_eval_episodes=config.val_eps)

    ### SETUP AND TRAIN
    # Setup model
    if config.MODEL_NAME == "A2C":
        model = A2C(config.POLICY, train_env, verbose=1, tensorboard_log=config.tb_path, seed=config.seed)
    elif config.MODEL_NAME == "PPO":
        model = PPO2(config.POLICY, train_env, verbose=1, tensorboard_log=config.tb_path, nminibatches=1, seed=config.seed)
    elif config.MODEL_NAME == "DDPG":
        # the noise objects for DDPG
        n_actions = train_env.action_space.shape[-1]
        param_noise = None
        action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))
        model = DDPG(config.POLICY, train_env, param_noise=param_noise, action_noise=action_noise, verbose=1, tensorboard_log=config.tb_path, seed=config.seed)
        print("DDPG does not provice training output...")

    ###
    # Train Model
    model = model.learn(total_timesteps=config.learn_steps, callback=eval_callback)

    # Load best model after training
    if config.MODEL_NAME == "A2C":
        model = A2C.load(load_path=config.val_path.joinpath("best_model.zip"))
    elif config.MODEL_NAME == "PPO":
        model = PPO2.load(load_path=config.val_path.joinpath("best_model.zip"))
    elif config.MODEL_NAME == "DDPG":
        model = DDPG.load(load_path=config.val_path.joinpath("best_model.zip"))

    ### EVAL MODEL
    # Make prediction in test_env
    test_mean, test_std = evaluate_policy(model=model, env=test_env, deterministic=config.deterministic,
                                n_eval_episodes=config.test_eps, return_episode_rewards=False)

    print(f"Test Mean:{test_mean}\n"+ \
          f"Test Std:{test_std}")

def random() -> None:
    ###
    # Setup ENV
    env = test_env

    ###
    # Demo loop
    for episode in range(config.test_eps):
        print(f"{episode+1}. Episode")
        _ = env.reset() # reset for each new episode
        done = False
        while not done: # run until done
            action = env.action_space.sample() # select a random action (see https://github.com/openai/gym/wiki/CartPole-v0)
            _, _, done, _ = env.step(action)
            if done:
                break

def BuyHold() -> None:
    ### PREPARATION
    # Which env do we want to use?
    env = test_env

    ep_rewards = []

    print(f"Start trading...")
    for episode in range(config.test_eps):
        state = env.reset()
        done = False
        ep_rewards.append([])
        while not done:
            action = buyHold(state[0], env.action_space)
            state, reward, done, _ = env.step([action])
            ep_rewards[episode].append(reward[0])
        print(f"{episode+1}. Episode reward: {sum(ep_rewards[episode])}")
        if (episode+1) % 10 == 0:
            sum_rewards = [sum(lst) for lst in ep_rewards]
            mean_reward = sum(sum_rewards) / len(sum_rewards)
            print(f"Mean reward after {episode+1}th Episode: {mean_reward}")

if __name__ == "__main__":
    if config.MODEL_NAME in config.drl_algos:
        DRL()
    elif config.MODEL_NAME == "BuyHold":
        BuyHold()
    elif config.MODEL_NAME == "Random":
        random()