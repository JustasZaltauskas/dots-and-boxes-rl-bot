import pyspiel
import DBNSymmetries


def getExampleState() -> pyspiel.DotsAndBoxesState:
    game_string = "dots_and_boxes(num_rows={},num_cols={})".format(2, 2)
    game: pyspiel.Game = pyspiel.load_game(game_string)
    return game.new_initial_state("000110011101")


def getExampleGame():
    game_string = "dots_and_boxes(num_rows={},num_cols={})".format(2, 2)
    return pyspiel.load_game(game_string)

def test():
    s = getExampleState()
    game = getExampleGame()

    symmetries = DBNSymmetries.getAllSymmetries(s.dbn_string(), (2, 2))

    # One can check visually if all states are symmetric
    for state in symmetries:
        print(game.new_initial_state(state))

    # We assert that all symmetries are different
    assert len(symmetries) == len(set(symmetries))
    print("All symmetries are different!")

    # Assert that a blank board has no symmetries
    assert len(set(DBNSymmetries.getAllSymmetries(game.new_initial_state().dbn_string(), (2, 2)))) == 1
    print("The blank board has no symmetries!")






def getExampleRectangleState():
    # ┌───┬───┬╴ ╶┐
    # │
    # ├╴ ╶┼╴ ╶┼───┤
    # │
    # └╴ ╶┴───┴╴ ╶┘
    game_string = "dots_and_boxes(num_rows={},num_cols={})".format(3, 2)
    game: pyspiel.Game = pyspiel.load_game(game_string)
    return game.new_initial_state("11000101001000010")


def rectStateTest():
    rect = getExampleRectangleState()
    symmetries = DBNSymmetries.getAllSymmetries(rect.dbn_string(), (3, 2))
    ss = DBNSymmetries._to_sparsed_matrix(rect.dbn_string(), (3,2))
    print(ss)


rectStateTest()


test()
