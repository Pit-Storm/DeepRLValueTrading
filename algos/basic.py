import config

def buyHold(state: list, action_space) -> list:
    """
    Basic Buy'n'Hold Agent.
    """
    # Check if we have Cash
    cash = state[0]
    if cash > 0:
        # Calculate how much money we can invest in each stock
        investing = cash / action_space.shape[0]

        # calculate num for each stock
        action = []
        for idx in range(action_space.shape[0]):
            price = state[1+action_space.shape[0]+idx]
            if price == 0:
                num = 0
            else:
                num = (investing // price) / config.ACTION_SCALING
                num = num if num <= 1 else 1
            action.append(num)
        
        return action