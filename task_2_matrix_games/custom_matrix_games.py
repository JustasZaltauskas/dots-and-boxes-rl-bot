"""
Implementations of matrix games according to the project description.
Some of these games are available in OpenSpiel, but may have different payoff tables.
"""

import pyspiel


def biased_rock_paper_scissors():
    """
    A zero-sum matrix game.

    https://en.wikipedia.org/wiki/Rock_paper_scissors
    """
    return pyspiel.create_matrix_game(
        "biased_rock_paper_scissors",
        "Biased Rock-Paper-Scissors",
        ["Rock", "Paper", "Scissors"],
        ["Rock", "Paper", "Scissors"],
        [[0, -0.25, 0.5], [0.25, 0, -0.05], [-0.5, 0.05, 0]],
        [[0, 0.25, -0.5], [-0.25, 0, 0.05], [0.5, -0.05, 0]],
    )


def dispersion():
    """
    A non-zero-sum coordination matrix game.

    https://aaai.org/papers/00398-dispersion-games-general-definitions-and-some-specific-learning-results/
    """
    return pyspiel.create_matrix_game(
        "dispersion",
        "Dispersion",
        ["A", "B"],
        ["A", "B"],
        [[-1, 1], [1, -1]],
        [[-1, 1], [1, -1]],
    )


def battle_of_the_sexes():
    """
    A non-zero-sum coordination matrix game.

    https://en.wikipedia.org/wiki/Battle_of_the_sexes_%28game_theory%29
    """
    return pyspiel.create_matrix_game(
        "battle_of_the_sexes",
        "Battle of the Sexes",
        ["Opera", "Movie"],
        ["Opera", "Movie"],
        [[3, 0], [0, 2]],
        [[2, 0], [0, 3]],
    )


def prisonners_dilemma():
    """
    A non-zero-sum social dilemma matrix game.

    https://en.wikipedia.org/wiki/Prisoner%27s_dilemma
    """
    return pyspiel.create_matrix_game(
        "prisoners_dilemma",
        "Prisoners' Dilemma",
        ["Confess", "Deny"],
        ["Confess", "Deny"],
        [[-1, -4], [0, -3]],
        [[-1, 0], [-4, -3]],
    )


ALL_CUSTOM_MATRIX_GAMES = [
    biased_rock_paper_scissors,
    dispersion,
    battle_of_the_sexes,
    prisonners_dilemma,
]
