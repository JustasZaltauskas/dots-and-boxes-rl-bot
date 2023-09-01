"""Lenient Boltzmann Q learning agent.

This implementation inherits the OpenSpiel Boltzmann Q-Learning implementation,
adding the "lenient" modification from the paper "Evolutionary Dynamics of
Multi-Agent Learning: A Survey" by Daan Bloembergen et al. (2015) 
(https://dl.acm.org/doi/10.5555/2831071.2831085).

The "lenient" modification keeps track of a moving average of past rewards in an 
"experience replay memory", then uses the memory to ignore occasional low rewards 
caused by exploration steps.
"""

import numpy as np

from open_spiel.python import rl_agent, rl_tools
from open_spiel.python.algorithms.boltzmann_tabular_qlearner import BoltzmannQLearner


class LenientBoltzmannQLearner(BoltzmannQLearner):
  def __init__(self,
               player_id,
               num_actions,
               step_size=0.1,
               discount_factor=1.0,
               temperature_schedule=rl_tools.ConstantSchedule(.5),
               # the amount of history in the ERM before Q-values get updated
               history_before_q=15,
               centralized=False):

    # initialize experience replay memory
    self._erm = {i:np.array([]) for i in range(num_actions)}
    self._history_before_q = history_before_q

    super().__init__(
        player_id,
        num_actions,
        step_size=step_size,
        discount_factor=discount_factor,
        temperature_schedule=temperature_schedule,
        centralized=centralized)

  def step(self, time_step, is_evaluation=False):
      """
      Mostly copied from the base class's implementation, with some modifications 
      for lenient Boltzmann behaviour.
      """
      if self._centralized:
          info_state = str(time_step.observations["info_state"])
      else:
          info_state = str(time_step.observations["info_state"][self._player_id])
      legal_actions = time_step.observations["legal_actions"][self._player_id]

      # Prevent undefined errors if this agent never plays until terminal step
      action, probs = None, None

      # Act step: don't act at terminal states.
      if not time_step.last():
          epsilon = 0.0 if is_evaluation else self._epsilon
          action, probs = self._get_action_probs(info_state, legal_actions, epsilon)

      # Learn step: don't learn during evaluation or at first agent steps.
      if self._prev_info_state and not is_evaluation:
          target = time_step.rewards[self._player_id]

          #### START: CUSTOMIZED SECTION
          # append the current reward to the experience replay memory
          self._erm[self._prev_action] = np.append(self._erm[self._prev_action], target)

          for action_key in self._erm:
              # only use/update experience replay memory once the history is adequate
              if self._erm[action_key].size >= self._history_before_q:

                  # leniency step: use the maximum reward in the experience replay memory as the target
                  target = self._erm[action_key].max(initial=-np.inf)

                  prev_q_value = self._q_values[self._prev_info_state][action_key]
                  self._last_loss_value = target - prev_q_value
                  self._q_values[self._prev_info_state][action_key] += (self._step_size * self._last_loss_value)
                  
                  # clear the experience replay memory
                  self._erm[action_key] = np.array([])

          ##### END: CUSTOMIZED SECTION

          if time_step.last():  # prepare for the next episode.
              self._prev_info_state = None
              return

      # Don't mess up with the state during evaluation.
      if not is_evaluation:
          self._prev_info_state = info_state
          self._prev_action = action
      return rl_agent.StepOutput(action=action, probs=probs)
