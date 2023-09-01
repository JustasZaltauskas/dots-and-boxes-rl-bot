"""
Engine to run several matrix games, where the players are agents using the same learning algorithm.

Sources:
- https://gitlab.kuleuven.be/dtai/courses/machine-learning-project/open_spiel/-/blob/dots_and_boxes/open_spiel/python/examples/dotsandboxes_example.py
- https://gitlab.kuleuven.be/dtai/courses/machine-learning-project/open_spiel/-/blob/dots_and_boxes/open_spiel/python/examples/independent_tabular_qlearning.py
"""

from custom_matrix_games import ALL_CUSTOM_MATRIX_GAMES
from lenient_boltzmann_q_learner import LenientBoltzmannQLearner

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
from open_spiel.python import rl_environment, rl_tools
from open_spiel.python.egt import visualization  # not directly used, but required for OpenSpiel dynamics plotting
from open_spiel.python.egt.utils import game_payoffs_array
from open_spiel.python.egt.dynamics import replicator, SinglePopulationDynamics, MultiPopulationDynamics
from open_spiel.python.algorithms.tabular_qlearner import QLearner
from open_spiel.python.algorithms.boltzmann_tabular_qlearner import BoltzmannQLearner

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "algorithm",
    "boltz",
    "Algorithm for the learner to use. Options: " +
    "eps (Epsilon-Greedy Tabular Q-Learner adapted from OpenSpiel), " +
    "boltz (OpenSpiel's Boltzmann Q-Learner), " + 
    "lenient_boltz (Lenient Boltzmann Q-Learning)",
)
flags.DEFINE_integer("num_train_episodes", 1000, "Number of training episodes.")
flags.DEFINE_integer(
    "num_eval_episodes", 10, "Number of episodes to use during each evaluation."
)
flags.DEFINE_integer(
    "eval_freq", 50, "The frequency (in episodes) to run evaluation."
)


def load_agent(algorithm: str, player_id: int, num_actions: int):
    if algorithm == "eps":
        return QLearner(
            player_id=player_id,
            num_actions=num_actions,
            epsilon_schedule=rl_tools.ConstantSchedule(0.2),
            step_size=1e-1,
            discount_factor=1)
    elif algorithm == "boltz":
        return BoltzmannQLearner(
            player_id=player_id,
            num_actions=num_actions,
            temperature_schedule=rl_tools.ConstantSchedule(0.2),
            step_size=1e-1,
            discount_factor=1)
    elif algorithm == "lenient_boltz":
        return LenientBoltzmannQLearner(
            player_id=player_id,
            num_actions=num_actions,
            temperature_schedule=rl_tools.ConstantSchedule(0.2),
            step_size=0.1,
            discount_factor=1)
    else:
        raise ValueError(f"Unknown algorithm {algorithm}")


def train_agents(env: rl_environment.Environment, agents: list, train_episodes: int):
    state_history = []

    for cur_episode in range(train_episodes):
        # Evaluate every FLAGS.eval_freq episodes.
        if cur_episode % int(FLAGS.eval_freq) == 0:
            avg_rewards = eval_agents(env, agents)
            print(f"Training episodes: {cur_episode}, Avg rewards: {avg_rewards}")

        # Play one episode.
        time_step = env.reset()
        while not time_step.last():
            agent_outputs = [agent.step(time_step) for agent in agents]
            state_history.append([agent_output.probs for agent_output in agent_outputs])
            time_step = env.step([agent_output.action for agent_output in agent_outputs])

        # Episode is over, step all agents with final info state.
        for agent in agents:
            agent.step(time_step)

    return state_history


def eval_agents(env: rl_environment.Environment, agents: list) -> np.array:
    """
    Evaluate the agents, returning a numpy array of average returns.
    """
    rewards = np.zeros(env.num_players, dtype=np.float64)

    for _ in range(FLAGS.num_eval_episodes):
        time_step = env.reset()
        
        while not time_step.last():
            agent_actions = [agent.step(time_step, is_evaluation=True).action for agent in agents]
            time_step = env.step(agent_actions)
        
        for i in range(env.num_players):
            rewards[i] += time_step.rewards[i]

    rewards /= FLAGS.num_eval_episodes
    return rewards


def draw_replicator_dynamics(games: list, algo_name: str):
    phase_fig = plt.figure(figsize=(len(games) * 6, 2 * 6))

    for game_idx, game in enumerate(games):
        game = game()
        game_name = game.get_type().long_name
        payoff_tensor = game_payoffs_array(game)
        projection_2x2 = game.num_cols() == 2
        projection = "2x2" if projection_2x2 else "3x3"

        # TODO adapt dynamics depending on algorithm (replicator, boltzmann, etc)
        # TODO adapt PopulationDynamics class for lenient case
        if projection_2x2:
            dyn = MultiPopulationDynamics(payoff_tensor, replicator)
            projection = "2x2"
            ax_labels = [f"P({game.row_action_name(0)}) Agent 1", f"P({game.row_action_name(0)}) Agent 2"]
        else:
            dyn = SinglePopulationDynamics(payoff_tensor, replicator)
            projection = "3x3"
            ax_labels = [game.row_action_name(0), game.row_action_name(1), game.row_action_name(2)]

        # Phase plots
        ax_streamline = plt.subplot2grid((2, len(games)), (0, game_idx), projection=projection)
        ax_quiver = plt.subplot2grid((2, len(games)), (1, game_idx), projection=projection)


        if projection_2x2:
            ax_streamline.set_xlabel(ax_labels[0])
            ax_quiver.set_xlabel(ax_labels[0])
            ax_streamline.set_ylabel(ax_labels[1])
            ax_quiver.set_ylabel(ax_labels[1])
        else:
            ax_streamline.set_labels(ax_labels)
            ax_quiver.set_labels(ax_labels)

        ax_streamline.streamplot(dyn, density=0.6, color='black') # , linewidth=1
        ax_quiver.quiver(dyn)

        ax_streamline.set_title(game_name)
        ax_quiver.set_title(game_name)

    plt.text(-4, 1.7, "Streamline", fontsize=20, horizontalalignment='left')
    plt.text(-4, 0.5, "Quiver", fontsize=20, horizontalalignment='left')
    phase_fig.savefig(f"images/phase_plots.png")

    #########################
    # Dynamics Plots
    #########################

    games_2x2 = [game() for game in games if game().num_cols() == 2]
    k_values = [1, 3, 5, 10, 20]
    dynamics_fig = plt.figure(figsize=(len(games_2x2) * 7.5, len(k_values) * 3))

    games_vis_dim = len(games_2x2) + 1
    k_vis_dim = len(k_values) + 1

    for game_idx, game in enumerate(games_2x2):
        game_name = game.get_type().long_name
        payoff_tensor = game_payoffs_array(game)

        ax = plt.subplot2grid((games_vis_dim, k_vis_dim), (game_idx, 0))
        plt.text(1, 0.5, game_name, fontsize=20, horizontalalignment='right')
        plt.axis('off')

        for k_idx, k in enumerate(k_values):
            # TODO change dynamics depending on agent algorithm
            dyn = MultiPopulationDynamics(payoff_tensor, replicator)
            ax = plt.subplot2grid((games_vis_dim, k_vis_dim), (game_idx, k_idx + 1), projection="2x2")
            ax.quiver(dyn)
            plt.xlabel(f"P({game.row_action_name(0)}) Agent 1")
            plt.ylabel(f"P({game.row_action_name(0)}) Agent 2")

    # bottom labels: values of k
    for k_idx, k in enumerate(k_values):
        ax = plt.subplot2grid((games_vis_dim, k_vis_dim), (games_vis_dim - 1, k_idx + 1))
        plt.text(0.5, 1, f"k = {k}", fontsize=20, verticalalignment='top', horizontalalignment='center')
        plt.axis('off')
    
    dynamics_fig.savefig(f"images/dynamics.png")


def trajectory_plot(game, population_histories, algo_name: str):
    game = game()
    game_name = game.get_type().long_name
    if game.num_cols() == 2:
        payoff_tensor = game_payoffs_array(game)

        # TODO change dynamics depending on agent algorithm
        dyn = MultiPopulationDynamics(payoff_tensor, replicator)
        
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="2x2")
        ax.quiver(dyn)

        for pop_history in population_histories:
            # Plot the probability agent 0 takes action 0 and agent 1 takes action 0
            ax.plot(pop_history[0][0], pop_history[1][0], 'ro')
        
        plt.title(game_name)
        plt.xlabel(f"P({game.row_action_name(0)}) Agent 1")
        plt.ylabel(f"P({game.row_action_name(0)}) Agent 2")
        plt.xlim(-0.01, 1.01)
        plt.ylim(-0.01, 1.01)
        plt.savefig(f"images/trajectory_{game.get_type().short_name}_{algo_name}.png")
    else:
        print(f"Skipping trajectory plot for {game_name} because it is not 2x2.")


def main(_):
    print("Drawing phase plots and replicator dynamics")
    draw_replicator_dynamics(ALL_CUSTOM_MATRIX_GAMES, FLAGS.algorithm)
    
    # Iterate over all the games to evaluate.
    print("Beginning training")
    
    for game in ALL_CUSTOM_MATRIX_GAMES:
        # Create the environment and agents.
        env = rl_environment.Environment(game())
        num_actions = env.action_spec()["num_actions"]
        agents = [
            load_agent(FLAGS.algorithm, player_id, num_actions)
            for player_id in range(env.num_players)
        ]

        # Train by playing the game for the specified number of episodes.
        population_histories = train_agents(env, agents, FLAGS.num_train_episodes)

        # plot learning trajectories
        # TODO place all on one figure
        print(f"Plotting learning trajectories for {game().get_type().long_name}")
        trajectory_plot(game, population_histories, FLAGS.algorithm)

if __name__ == "__main__":
    app.run(main)
