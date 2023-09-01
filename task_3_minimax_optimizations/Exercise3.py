import datetime

import pyspiel
import DBNUtils
import DBNSymmetries

def main():
    """
    Runs the code for exercise 3.
    This method solves the dots and boxes game for small grids and plots the relevant results
    """
    file = open("results.txt", "a")
    file.writelines(str(datetime.datetime.now())+"\n\n")
    description = "Each entry represents the time spent, number of states searched and size of transposition table (if performed by an optimized searcher)\n"
    print(description)
    file.writelines(description)

    dimensions = [(1, 1), (1, 2), (1, 3), (2, 2)]
    solvers = [
    #    (getNaiveMinimaxSolverFor, "NaiveSolver"),
        (getHalfOptimisedMinimaxSolverFor, "HalfOptimised"),
        (getOptimisedMinimaxSolverFor, "FullyOptimised")
    ]

    for dim in dimensions:
        description = "Starting run for dimension " + str(dim) + "\n"
        file.writelines(description)
        print(description)

        for solverFn, name in solvers:
            solver = solverFn(dim)
            runResults = solver.solveGame(dim)
            dispStr = "{}: {}, N = {}\n".format(name, runResults, solver.getTranspositionTableSize())
            print(dispStr)
            file.writelines(dispStr)

        file.writelines("\n\n")

    # End of run
    file.writelines(str(datetime.datetime.now())+"\n\n")


def getInitialStateOfSize(nbRows, nbCols) -> pyspiel.DotsAndBoxesState:
    """
    Returns a new dots and boxes state of the given size
    """
    game_string = "dots_and_boxes(num_rows={},num_cols={})".format(nbRows, nbCols)
    return pyspiel.load_game(game_string).new_initial_state()

def getOptimisedMinimaxSolverFor(dbnDim):
    return OptimizedMinimax(OptimisedTranspositionTable(dbnDim))

def getHalfOptimisedMinimaxSolverFor(dbnDim):
    return OptimizedMinimax(NaiveTranspositionTable(dbnDim))

def getNaiveMinimaxSolverFor(dbnDim):
    return NaiveMinimax()

class GameState(object):
    """
    A class compactly representing what the game state is as to only entail the relevant features.
    This means it only keeps track of the dbn-string and the score of the first player
    """
    def __init__(self, dbnString, scorePlayerOne):
        self.dbnString = dbnString
        self.scorePlayerOne = scorePlayerOne

    def FromDBNState(dbnState: pyspiel.DotsAndBoxesState):
        return GameState(dbnState.dbn_string(), DBNUtils.getScoreOfFirstPlayerOnState(dbnState))

    def __eq__(self, other):
        return self.dbnString.__eq__(other.dbnString) & self.scorePlayerOne == other.scorePlayerOne

    def __hash__(self):
        return hash((self.dbnString, self.scorePlayerOne))

    def getSymmetricalGameStates(self, gameDim):
        """
        :param gameDim: the dimension of the game
        :return: A list of all the game states that are symmetrical to this one
        """
        return [GameState(symStr, self.scorePlayerOne) for symStr in
                DBNSymmetries.getAllSymmetries(self.dbnString, gameDim)]

class OptimisedTranspositionTable(object):
    """
    A class representing a transposition table who is able to reason about symmetries
    """

    def __init__(self, gameDim):
        self.entries = dict()
        self.gameDim = gameDim

    def getEvaluationOfState(self, state: GameState) -> int:
        """
        Returns the evaluation of (a symetry of) the given state, or None if nothing was found (not saved in this table yet)
        """
        symmetricals = state.getSymmetricalGameStates(self.gameDim)
        for x in symmetricals:
            entry = self.entries.get(x)
            if entry is not None:
                return entry
        return None
        # return next((self.entries.get(x) for x in symmetricals), None)

    def insertStateEvaluation(self, state: GameState, evaluation) -> None:
        """
        Saves the given game state and its evaluation in this table
        """
        self.entries[state] = evaluation

    def getNumberOfItems(self):
        """
        :return: the number of states saved in this table
        """
        return self.entries.__len__()


class NaiveTranspositionTable(object):
    """
    A class representing a transposition table
    """

    def __init__(self, gameDim):
        self.entries = dict()
        self.gameDim = gameDim

    def getEvaluationOfState(self, state: GameState) -> int:
        """
        Returns the evaluation of (a symetry of) the given state, or None if nothing was found (not saved in this table yet)
        """
        return self.entries.get(state)

    def insertStateEvaluation(self, state: GameState, evaluation) -> None:
        """
        Saves the given game state and its evaluation in this table
        """
        self.entries[state] = evaluation

    def getNumberOfItems(self):
        """
        :return: the number of states saved in this table
        """
        return self.entries.__len__()


class MinimaxSolver(object):
    def __init__(self):
        self.nbStatesVisited = 0

    def solveGame(self, dbnDim):
        """
        Solves the dots and boxes game and returns the number of states visited and the running time
        """
        initial_state = getInitialStateOfSize(dbnDim[0], dbnDim[1])
        self.nbStatesVisited = 0
        start_time = datetime.datetime.now()
        self.minimax(initial_state, initial_state.current_player())
        end_time = datetime.datetime.now()
        time_spent = end_time - start_time
        return [time_spent, self.nbStatesVisited]

    def minimax(self, state, maximizing_player_id):
        """
        Evaluates the value of the state for the player_id by performing minimax.
        :returns: The evaluation of the state
        """
        self.nbStatesVisited += 1

class NaiveMinimax(MinimaxSolver):
    def minimax(self, state, maximizing_player_id):
        super().minimax(state, maximizing_player_id)
        if state.is_terminal():
            return state.player_return(maximizing_player_id)

        player = state.current_player()
        if player == maximizing_player_id:
            selection = max
        else:
            selection = min
        values_children = [self.minimax(state.child(action), maximizing_player_id) for action in state.legal_actions()]
        return selection(values_children)

    def getTranspositionTableSize(self):
        return None

class OptimizedMinimax(MinimaxSolver):
    def __init__(self, transpositionTable):
        super().__init__()
        self.transpositionTable = transpositionTable

    def getTranspositionTableSize(self) -> int:
        return len(self.transpositionTable.entries)


    def minimax(self, state, maximizing_player_id):
        super().minimax(state, maximizing_player_id)
        if state.is_terminal():
            return state.player_return(maximizing_player_id)

        GS = GameState.FromDBNState(state)
        # Check if this state has already been evaluated
        saved_eval = self.transpositionTable.getEvaluationOfState(GS)
        if saved_eval is not None:
            return saved_eval

        player = state.current_player()
        if player == maximizing_player_id:
            selection = max
        else:
            selection = min
        values_children = [self.minimax(state.child(action), maximizing_player_id) for action in state.legal_actions()]
        res = selection(values_children)

        # Insert the state into the transposition table
        self.transpositionTable.insertStateEvaluation(GS, res)
        return res


main()