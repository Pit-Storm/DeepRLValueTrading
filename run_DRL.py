from data import handling as dth
from gymEnv.valueTrading import valueTradingEnv
import config


def main() -> None:
    ###
    # Define variables

    ###
    # Load Dataset
    stocks_df = dth.load_data("data/stocksdata_all.csv")

    # make train, val, test df
    train, val, test = dth.train_val_test_split(stocks_df)

    ###
    # Setup ENV
    trading_env = valueTradingEnv(train)

    ###
    # Demo loop
    for episode in range(2):
        state, info = trading_env.reset() # reset for each new trial
        for t in range(30000): # run for n timesteps or until done, whichever is first
            # trading_env.render()
            print(f"{t+1}. Step in {episode+1}. Episode")
            action = trading_env.action_space.sample() # select a random action (see https://github.com/openai/gym/wiki/CartPole-v0)
            state, reward, done, info = trading_env.step(action)
            print(f"Date: {trading_env.date} | End: {trading_env.end_date} | Cash: {info['cashes'][-1]}")
            print(f"Open: {info['openPrices'][-1][0]} | Close: {info['closePrices'][-1][0]} | Action: {info['actions'][-1][0]}")
            if done:
                print(f"Done!")
                # print("="*15)
                break
        print("="*15)

    ###
    # Trainingloop

        ###
        # Train Model

        ###
        # Validate Training every n steps

        ###
        # If model has validation measurement over specific
        # threshold stop training

    ###
    # 

if __name__ == "__main__":
    main()
