import config

def buyHold(state: list, action_space) -> list:
    """
    Basic Buy'n'Hold Agent.
    
    It divides the available cash by the num of symbols. If it is possible to buy a stock with the fraction of cash it will buy the possible number of shares.
    """
    # Check if we have Cash
    cash = state[0]
    # Calculate how much money we can invest in each stock
    investing = cash / action_space.shape[0]

    # calculate num for each stock
    action = []
    for idx in range(action_space.shape[0]):
        price = state[1+action_space.shape[0]+idx]
        # We need this to avoid division by zero for stocks not tradable at the given state.
        if price == 0:
            num = 0
        else:
            num = (investing // price) / config.ACTION_SCALING
            num = num if num <= 1 else 1
        action.append(num)
        
        return action