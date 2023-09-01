import pyspiel

def getScoreOfState(dbnState: pyspiel.DotsAndBoxesState):
    """
    Returns the score of the game

    # ┌╴ ╶┬───┐
    #     │ 2 │
    # ├╴ ╶┼───┤
    # │ 1 │ 1 │
    # └───┴───┘
    Example: Above board will return (2,1)
    """
    obs_str = dbnState.observation_string()
    return [obs_str.count("1"), obs_str.count("2")]

def getScoreOfFirstPlayerOnState(state):
    """
    Returns the score of the game for the first player

    # ┌╴ ╶┬───┐
    #     │ 2 │
    # ├╴ ╶┼───┤
    # │ 1 │ 1 │
    # └───┴───┘
    Example: Above board will return 2
    """
    return state.observation_string().count("1")

