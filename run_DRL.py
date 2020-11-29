from data import handling as dth
from gymEnv.valueTrading import valueTradingEnv
import config
from algos.basic import buyHold
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.evaluation import evaluate_policy
from pathlib import Path
from datetime import datetime
import pandas as pd
from os import system

system("clear")

### VARS
yearrange = 4
episodic = False
train_envs = 1
val_envs = 1
val_freq = 10000
val_eps = 5
test_eps = 100
learn_steps = int(6 * val_freq)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
base_path = config.BASE_PATH / config.MODEL_NAME / timestamp
env_path = base_path / config.ENV_INFO_PATH
tb_path = base_path / config.TB_LOGS_PATH
best_path = base_path / config.BEST_MODELS_PATH
data_path = Path.cwd().joinpath("data","stocksdata_all.csv")

def main() -> None:
    ### DATA
    # Load Dataset
    stocks_df = dth.load_data(data_path)

    # make train, val, test df
    train, val, test = dth.train_val_test_split(stocks_df)

    ### PREPARATION
    # Training Env
    train_env = DummyVecEnv([lambda: valueTradingEnv(df=train, episodic=episodic, yearrange=yearrange,
                            save_path=env_path.joinpath("train")) for i in range(train_envs)])
    train_env = VecCheckNan(train_env, raise_exception=True)
    # train_env = VecNormalize(train_env)
    # Validation Env
    val_env = DummyVecEnv([lambda: valueTradingEnv(df=val, train=False, episodic=episodic, yearrange=yearrange,
                            save_path=env_path.joinpath("val")) for i in range(val_envs)])
    val_env = VecCheckNan(val_env, raise_exception=True)
    # val_env = VecNormalize(val_env)
    # test_env
    test_env = DummyVecEnv([lambda: valueTradingEnv(df=test, train=False, episodic=episodic, yearrange=yearrange,
                            save_path=env_path.joinpath("test"))])
    test_env = VecCheckNan(test_env, raise_exception=True)
    # test_env = VecNormalize(test_env)

    # callback for validation
    eval_callback = EvalCallback(val_env, best_model_save_path=best_path,
                             log_path=tb_path, eval_freq=val_freq,
                             deterministic=True, n_eval_episodes=val_eps)

    ### SETUP AND TRAIN
    # Setup model
    model = A2C('MlpLstmPolicy', train_env, verbose=1, tensorboard_log=tb_path)
    ###
    # Train Model
    model = model.learn(total_timesteps=learn_steps, callback=eval_callback)

    ### TODO
    # If model has validation measurement over specific
    # threshold stop training

    ### EVAL MODEL
    # Make prediction in test_env
    test_mean, test_std = evaluate_policy(model=model, env=test_env,
                                n_eval_episodes=test_eps, return_episode_rewards=False)

    print(f"Test Mean:{test_mean}\n"+ \
          f"Test Std:{test_std}")

def random() -> None:
    ### VARS
    test_eps = 100

    ###
    # Load Dataset
    stocks_df = dth.load_data(data_path)

    # make train, val, test df
    _, _, test = dth.train_val_test_split(stocks_df)

    ###
    # Setup ENV
    test_env = valueTradingEnv(test, train=False, save_path=env_path)

    ###
    # Demo loop
    for episode in range(test_eps):
        print(f"{episode+1}. Episode")
        _ = test_env.reset() # reset for each new episode
        done = False
        while not done: # run until done
            action = test_env.action_space.sample() # select a random action (see https://github.com/openai/gym/wiki/CartPole-v0)
            _, _, done, _ = test_env.step(action)
            if done:
                break

def buyAndHold() -> None:
    ### VARS
    test_eps = 1000
    episodic = False

    ### DATA
    # Load Dataset
    stocks_df = dth.load_data(data_path)

    # make train, val, test df
    train, val, test = dth.train_val_test_split(stocks_df)

    ### PREPARATION
    # Training Env
    train_env = DummyVecEnv([lambda: valueTradingEnv(df=train, episodic=episodic, yearrange=yearrange) for i in range(train_envs)])

    # Validation Env
    val_env = DummyVecEnv([lambda: valueTradingEnv(df=val, train=False, episodic=episodic, yearrange=yearrange) for i in range(val_envs)])

    # test_env
    test_env = DummyVecEnv([lambda: valueTradingEnv(df=test, train=False, episodic=episodic, yearrange=yearrange)])
    
    ep_rewards = []

    print(f"Start trading...")
    for episode in range(test_eps):
        state = train_env.reset()
        done = False
        ep_rewards.append([])
        while not done:
            action = buyHold(state[0], train_env.action_space)
            state, reward, done, _ = train_env.step([action])
            ep_rewards[episode].append(reward[0])
        print(f"{episode+1}. Episode reward: {sum(ep_rewards[episode])}")
        if (episode+1) % 10 == 0:
            sum_rewards = [sum(lst) for lst in ep_rewards]
            mean_reward = sum(sum_rewards) / len(sum_rewards)
            print(f"Mean reward after {episode+1}th Episode: {mean_reward}")

    # rewards_df = pd.DataFrame(ep_rewards, columns=range(eps))
    # rewards_mean = rewards_df.mean()
    # rewards_std = rewards_df.std()

    # print(f"Buy and Hold Algo")
    # print(f"Mean reward: {rewards_mean} | Std reward {rewards_std}")

def test() -> None:
    print()

if __name__ == "__main__":
    # main()
    # random()
    buyAndHold()
    # test()
