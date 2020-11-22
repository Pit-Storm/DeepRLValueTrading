from data import handling as dth
from gymEnv.valueTrading import valueTradingEnv
import config
from stable_baselines import A2C
from stable_baselines.common.policies import MlpPolicy, MlpLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecCheckNan, VecNormalize
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.evaluation import evaluate_policy
from os import path
from datetime import datetime

def main() -> None:
    ###
    # Define variables
    train_envs = 1
    val_envs = 1
    val_freq = 4000
    val_eps = 5
    test_eps = 100
    a2c_steps = 20000
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
    a2c_tb_path = path.join(config.A2C_TB_PATH, timestamp)
    a2c_best_path = path.join(config.A2C_MODEL_PATH,timestamp)

    ###
    # Load Dataset
    stocks_df = dth.load_data("data/stocksdata_all.csv")

    # make train, val, test df
    train, val, test = dth.train_val_test_split(stocks_df)

    ###
    # Training Env
    train_env = DummyVecEnv([lambda: valueTradingEnv(df=train, train=True) for i in range(train_envs)])
    train_env = VecCheckNan(train_env, raise_exception=True)
    train_env = VecNormalize(train_env)
    # Validation Env
    val_env = DummyVecEnv([lambda: valueTradingEnv(df=val, train=False) for i in range(val_envs)])
    val_env = VecCheckNan(val_env, raise_exception=True)
    val_env = VecNormalize(val_env)
    # test_env
    test_env = DummyVecEnv([lambda: valueTradingEnv(df=test, train=False)])
    val_env = VecCheckNan(val_env, raise_exception=True)
    val_env = VecNormalize(val_env)
    # test_env = VecNormalize(test_env)

    ###
    # Setup model
    a2c_model = A2C('MlpLstmPolicy', train_env, verbose=1, tensorboard_log=a2c_tb_path)
    # callback for validation
    eval_callback = EvalCallback(val_env, best_model_save_path=a2c_best_path,
                             log_path=a2c_tb_path, eval_freq=val_freq,
                             deterministic=True, n_eval_episodes=val_eps)

    ###
    # Train Model
    a2c_model = a2c_model.learn(total_timesteps=a2c_steps, callback=eval_callback)

    ### TODO
    # If model has validation measurement over specific
    # threshold stop training

    ###
    # Make prediction in test_env
    test_mean, test_rewards = evaluate_policy(model=a2c_model, env=test_env,
                                n_eval_episodes=test_eps, return_episode_rewards=False)

    print(f"Test Mean:{test_mean}\n"+ \
          f"Test Rewards:{test_rewards}")

def demo() -> None:
    ###
    # Load Dataset
    stocks_df = dth.load_data("data/stocksdata_all.csv")

    # make train, val, test df
    train, *_ = dth.train_val_test_split(stocks_df)

    ###
    # Setup ENV
    trading_env = valueTradingEnv(train)

    ###
    # Demo loop
    for episode in range(2):
        _, info = trading_env.reset() # reset for each new trial
        for t in range(30000): # run for n timesteps or until done, whichever is first
            # trading_env.render()
            print(f"{t+1}. Step in {episode+1}. Episode")
            action = trading_env.action_space.sample() # select a random action (see https://github.com/openai/gym/wiki/CartPole-v0)
            _, _, done, info = trading_env.step(action)
            print(f"Date: {trading_env.date} | End: {trading_env.end_date} | Cash: {info['cashes'][-1]}")
            print(f"Open: {info['openPrices'][-1][0]} | Close: {info['closePrices'][-1][0]} | Action: {info['actions'][-1][0]}")
            if done:
                print(f"Done!")
                # print("="*15)
                break
        print("="*15)

def test() -> None:
    ###
    # Load Dataset
    stocks_df = dth.load_data("data/stocksdata_all.csv")

    # make train, val, test df
    train, *_ = dth.train_val_test_split(stocks_df)

    ###
    # Setup ENV
    trading_env = valueTradingEnv(train)

    ###
    # Testings
    obs = trading_env.reset()
    print(f"Len: {len(obs)} | dtype: {type(obs)}")
    print(obs[0])
    print(obs[1])

if __name__ == "__main__":
    main()
    # demo()
    # test()
