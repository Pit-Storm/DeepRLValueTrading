from data import handling as dth
from gymEnv.valueTrading import valueTradingEnv
import config
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.evaluation import evaluate_policy
from pathlib import Path
from datetime import datetime

timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
base_path = config.BASE_PATH / timestamp / config.MODEL_NAME
env_path = base_path / config.ENV_INFO_PATH
data_path = Path.cwd().joinpath("data","stocksdata_all.csv")


def main() -> None:
    ###
    # Define variables
    train_envs = 1
    val_envs = 1
    val_freq = 4000
    val_eps = 5
    test_eps = 100
    learn_steps = 50000
    tb_path = base_path / config.TB_LOGS_PATH
    best_path = base_path / config.BEST_MODELS_PATH

    ### DATA
    # Load Dataset
    stocks_df = dth.load_data(data_path)

    # make train, val, test df
    train, val, test = dth.train_val_test_split(stocks_df)

    ### PREPARATION
    # Training Env
    train_env = DummyVecEnv([lambda: valueTradingEnv(df=train, train=True, save_path=env_path.joinpath("train")) for i in range(train_envs)])
    # train_env = VecCheckNan(train_env, raise_exception=True)
    # train_env = VecNormalize(train_env)
    # Validation Env
    val_env = DummyVecEnv([lambda: valueTradingEnv(df=val, train=False, save_path=env_path.joinpath("val")) for i in range(val_envs)])
    # val_env = VecCheckNan(val_env, raise_exception=True)
    # val_env = VecNormalize(val_env)
    # test_env
    test_env = DummyVecEnv([lambda: valueTradingEnv(df=test, train=False, save_path=env_path.joinpath("test"))])
    # test_env = VecCheckNan(test_env, raise_exception=True)
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
    test_mean, test_rewards = evaluate_policy(model=model, env=test_env,
                                n_eval_episodes=test_eps, return_episode_rewards=False)

    print(f"Test Mean:{test_mean}\n"+ \
          f"Test Rewards:{test_rewards}")

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

if __name__ == "__main__":
    # main()
    random()
    # test()
