#!/usr/bin/env python3
# encoding: utf-8
"""
dotsandboxes_agent.py

Extend this class to provide an agent that can participate in a tournament.

Created by Pieter Robberechts, Wannes Meert.
Copyright (c) 2022 KU Leuven. All rights reserved.
"""

import sys
import logging
import numpy as np
import time

import pyspiel
from open_spiel.python.algorithms import evaluate_bots, mcts


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
        """
        pyspiel.Bot.__init__(self)
        self.player_id = player_id

        # MCTS hyper-parameters from open_spiel/python/examples/mcts.py
        self.uct_c = 2
        self._random_state = np.random.RandomState(seed=SEED)
        self.evaluator = mcts.RandomRolloutEvaluator(n_rollouts=1, random_state=self._random_state)
        self.verbose = True
        self.solve = True
        self.max_utility = pyspiel.load_game("dots_and_boxes").max_utility()
        self._dirichlet_noise = None
        self._child_selection_fn = mcts.SearchNode.uct_value
        self.dont_return_chance_node = False

        # STUDENT CUSTOMIZATION: bound simulation by time instead of iterations
        # Tournament is time-bounded and board size is variable,
        # bounding by time is safer and more optimal than iterations.
        # self.max_simulations = 45
        self.max_simulation_runtime_ms = 1_000

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
        """
        return self.step_with_policy(state)[1]
    
    def step_with_policy(self, state):
        """
        Source: open_spiel/python/algorithms/mcts.py
        """
        t1 = time.time()
        root = self.mcts_search(state)

        best = root.best_child()

        if self.verbose:
            seconds = time.time() - t1
            print("Finished {} sims in {:.3f} secs, {:.1f} sims/s".format(
                root.explore_count, seconds, root.explore_count / seconds))
            print("Root:")
            print(root.to_str(state))
            print("Children:")
            print(root.children_str(state))
            if best.children:
                chosen_state = state.clone()
                chosen_state.apply_action(best.action)
                print("Children of chosen:")
                print(best.children_str(chosen_state))

        mcts_action = best.action

        policy = [(action, (1.0 if action == mcts_action else 0.0))
                for action in state.legal_actions(state.current_player())]

        return policy, mcts_action

    def _apply_tree_policy(self, root, state):
        """
        Source: open_spiel/python/algorithms/mcts.py
        """
        visit_path = [root]
        working_state = state.clone()
        current_node = root
        while (not working_state.is_terminal() and
            current_node.explore_count > 0) or (
                working_state.is_chance_node() and self.dont_return_chance_node):
            if not current_node.children:
                # For a new node, initialize its state, then choose a child as normal.
                legal_actions = self.evaluator.prior(working_state)
                if current_node is root and self._dirichlet_noise:
                    epsilon, alpha = self._dirichlet_noise
                    noise = self._random_state.dirichlet([alpha] * len(legal_actions))
                    legal_actions = [(a, (1 - epsilon) * p + epsilon * n)
                                    for (a, p), n in zip(legal_actions, noise)]
                # Reduce bias from move generation order.
                self._random_state.shuffle(legal_actions)
                player = working_state.current_player()
                current_node.children = [
                    mcts.SearchNode(action, player, prior) for action, prior in legal_actions
                ]

            if working_state.is_chance_node():
                # For chance nodes, rollout according to chance node's probability
                # distribution
                outcomes = working_state.chance_outcomes()
                action_list, prob_list = zip(*outcomes)
                action = self._random_state.choice(action_list, p=prob_list)
                chosen_child = next(
                    c for c in current_node.children if c.action == action)
            else:
                # Otherwise choose node with largest UCT value
                chosen_child = max(
                    current_node.children,
                    key=lambda c: self._child_selection_fn(  # pylint: disable=g-long-lambda
                        c, current_node.explore_count, self.uct_c))

            working_state.apply_action(chosen_child.action)
            current_node = chosen_child
            visit_path.append(current_node)

        return visit_path, working_state

    def mcts_search(self, state):
        """
        Source: open_spiel/python/algorithms/mcts.py

        STUDENT CUSTOMIZATION: bound simulation by time instead of iterations
        """
        simulation_start_time = time.time()
        root = mcts.SearchNode(None, state.current_player(), 1)
        while (time.time() - simulation_start_time) * 1000 < self.max_simulation_runtime_ms:
            visit_path, working_state = self._apply_tree_policy(root, state)
            if working_state.is_terminal():
                returns = working_state.returns()
                visit_path[-1].outcome = returns
                solved = self.solve
            else:
                returns = self.evaluator.evaluate(working_state)
                solved = False

            while visit_path:
                # For chance nodes, walk up the tree to find the decision-maker.
                decision_node_idx = -1
                while visit_path[decision_node_idx].player == pyspiel.PlayerId.CHANCE:
                    decision_node_idx -= 1
                # Chance node targets are for the respective decision-maker.
                target_return = returns[visit_path[decision_node_idx].player]
                node = visit_path.pop()
                node.total_reward += target_return
                node.explore_count += 1

                if solved and node.children:
                    player = node.children[0].player
                    if player == pyspiel.PlayerId.CHANCE:
                        # Only back up chance nodes if all have the same outcome.
                        # An alternative would be to back up the weighted average of
                        # outcomes if all children are solved, but that is less clear.
                        outcome = node.children[0].outcome
                        if (outcome is not None and
                            all(np.array_equal(c.outcome, outcome) for c in node.children)):
                            node.outcome = outcome
                        else:
                            solved = False
                    else:
                        # If any have max utility (won?), or all children are solved,
                        # choose the one best for the player choosing.
                        best = None
                        all_solved = True
                        for child in node.children:
                            if child.outcome is None:
                                all_solved = False
                            elif best is None or child.outcome[player] > best.outcome[player]:
                                best = child
                        if (best is not None and
                            (all_solved or best.outcome[player] == self.max_utility)):
                            node.outcome = best.outcome
                        else:
                            solved = False
            if root.outcome is not None:
                break

        return root


def test_api_calls():
    """This method calls a number of API calls that are required for the
    tournament. It should not trigger any Exceptions.
    """
    dotsandboxes_game_string = (
        "dots_and_boxes(num_rows=5,num_cols=5)")
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

