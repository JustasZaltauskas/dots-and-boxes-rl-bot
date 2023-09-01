#!/usr/bin/env python3
# encoding: utf-8
"""
An agent implementing the AlphaZero algorithm to participate in a tournament
of the Dots and Boxes game.

Template Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.

Some sources:
- https://github.com/suragnair/alpha-zero-general/blob/master/Arena.py
- https://openspiel.readthedocs.io/en/latest/alpha_zero.html
- open_spiel/python/examples/mcts.py
"""

import os
import sys
import logging
import random
import numpy as np
import pyspiel
from open_spiel.python.algorithms import evaluate_bots, mcts
from open_spiel.python.algorithms.alpha_zero import evaluator as az_evaluator

# open_spiel.python.algorithms.alpha_zero.model is used for training,
# but it depends on Tensorflow v1, which can't be used in tournament play
import alpha_zero_inference_model as az_model


logger = logging.getLogger('be.kuleuven.cs.dtai.dotsandboxes')
SEED = 1234


def get_agent_for_tournament(player_id):
    """Change this function to initialize your agent.
    This function is called by the tournament code at the beginning of the
    tournament.

    :param player_id: The integer id of the player for this bot, e.g. `0` if
        acting as the first player.
    """
    my_player = Agent(player_id)
    return my_player


class Agent(pyspiel.Bot):
    """Agent template"""

    def __init__(self, player_id):
        """Initialize an agent to play Dots and Boxes.

        Note: This agent should make use of a pre-trained policy to enter
        the tournament. Initializing the agent should thus take no more than
        a few seconds.

        TODO see alpha_zero.evaluator() for cleaner instantiation
        """
        pyspiel.Bot.__init__(self)
        self.player_id = player_id

        # TODO make adaptations for multiple board sizes
        package_directory = os.path.dirname(os.path.abspath(__file__))
        checkpoint = os.path.join(package_directory, '../az_checkpoints/dots_and_boxes_7x7_actors4_sim75/checkpoint-350')
        model = az_model.Model.from_checkpoint(checkpoint)
        
        # TODO implement children of the MCTSBot and AlphaZeroEvaluator classes
        #      so we don't need a game instance here
        game = pyspiel.load_game('dots_and_boxes')

        evaluator = az_evaluator.AlphaZeroEvaluator(game, model)

        # TODO custom implementation that doesn't need game object
        # source: alpha_zero._init_bot()
        self.agent = mcts.MCTSBot(
            game,
            uct_c=2,
            max_simulations=10,
            evaluator=evaluator,
            solve=False,
            dirichlet_noise=None,
            child_selection_fn=mcts.SearchNode.puct_value,
            verbose=False,
            dont_return_chance_node=True)

    def restart_at(self, state):
        """Starting a new game in the given state.

        :param state: The initial state of the game.
        """
        pass

    def inform_action(self, state, player_id, action):
        """Let the bot know of the other agent's actions.

        :param state: The current state of the game.
        :param player_id: The ID of the player that executed an action.
        :param action: The action which the player executed.
        """
        pass

    def step(self, state):
        """Returns the selected action in the given state.

        :param state: The current state of the game.
        :returns: The selected action from the legal actions, or
            `pyspiel.INVALID_ACTION` if there are no legal actions available.
        
        TODO cite source: alpha_zero._play_game()
        """
        root = self.agent.mcts_search(state)
        policy = np.zeros(state.num_distinct_actions())

        for c in root.children:
            policy[c.action] = c.explore_count
        policy /= policy.sum()

        action = root.best_child().action
        return action


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=7,num_cols=7)")
    game = pyspiel.load_game(dotsandboxes_game_string)
    bots = [get_agent_for_tournament(player_id) for player_id in [0,1]]
    returns = evaluate_bots.evaluate_bots(game.new_initial_state(), bots, np.random)
    assert len(returns) == 2
    assert isinstance(returns[0], float)
    assert isinstance(returns[1], float)
    print("SUCCESS!")


def main(argv=None):
    test_api_calls()


if __name__ == "__main__":
    sys.exit(main())
