# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent components for planning."""
import re

from collections.abc import Sequence
import datetime
from typing import Callable
from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class SimPlan(component.Component):
  """Component representing the agent's plan."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      components: list[component.Component],
      clock_now: Callable[[], datetime.datetime],
      goal: component.Component | None = None,
      num_memories_to_retrieve: int = 5,
      timescale: str = 'the rest of the day',
      time_adverb: str = 'hourly',
      verbose: bool = False,
      log_color='green',
  ):
    """Initialize a component to represent the agent's plan.

    Args:
      model: a language model
      memory: an associative memory
      agent_name: the name of the agent
      components: components to build the context of planning
      clock_now: time callback to use for the state.
      goal: a component to represent the goal of planning
      num_memories_to_retrieve: how many memories to retrieve as conditioning
        for the planning chain of thought
      timescale: string describing how long the plan should last
      time_adverb: string describing the rate of steps in the plan
      verbose: whether or not to print intermediate reasoning stepst
      log_color: color for debug logging
    """
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._log_color = log_color
    self._components = components
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._goal_component = goal
    self._timescale = timescale
    self._time_adverb = time_adverb
    self._clock_now = clock_now
    self._last_update = datetime.datetime.min

    self._latest_memories = ''
    self._last_observation = []
    self._current_plan = ''
    self._history = []

    self._verbose = verbose

  def name(self) -> str:
    return 'Plan'

  def state(self):
    return self._state

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def _log(self, entry: str):
    print(termcolor.colored(entry, self._log_color), end='')

  def observe(self, observation: str):
    self._last_observation.append(observation)

  def get_components(self) -> Sequence[component.Component]:
    return self._components

  def update(self):
    if self._last_update == self._clock_now():
      return
    self._last_update = self._clock_now()

    observation = '\n'.join(self._last_observation)
    self._last_observation = []
    memories = self._memory.retrieve_associative(
        observation,
        k=self._num_memories_to_retrieve,
        use_recency=True,
        add_time=True,
    )
    if self._goal_component:
      memories = memories + self._memory.retrieve_associative(
          self._goal_component.state(),
          k=self._num_memories_to_retrieve,
          use_recency=True,
          add_time=True,
      )
      
    memories = '\n'.join(memories)

    components = '\n'.join([
        f"{self._agent_name}'s {construct.name()}:\n{construct.state()}"
        for construct in self._components
    ])

    in_context_example = (
        ' Please format the plan like in this example: [21:00 - 22:00] watch TV'
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'{components}\n')
    prompt.statement(f'Relevant memories:\n{memories}')
    if self._goal_component:
      prompt.statement(f'Current goal: {self._goal_component.state()}.')
    prompt.statement(f'Current situation: {observation}')

    time_now = self._clock_now().strftime('[%d %b %Y %H:%M:%S]')
    prompt.statement(f'The current time is: {time_now}\n')

    
    if re.search(r'(?<=current goal is )(.*?)(?=\.)', self._goal_component.state()) is not None:
      goal_mention = re.search(r'(?<=current goal is )(.*?)(?=\.)', self._goal_component.state()).group(1)
    else:
      goal_mention = self._goal_component.state()
    self._current_plan = prompt.open_question(
        f"Rewrite {self._agent_name}'s goal, {goal_mention}, "
        f"in terms of a single concrete action that can be taken "
        f"or started in the next five minutes.",
        max_characters=1200,
        max_tokens=1200,
        terminators=(),
    )

    self._state = self._current_plan

    if self._verbose:
      self._log('\n' + prompt.view().text() + '\n')

    update_log = {
        'Summary': (
            f'{self._time_adverb} plan of {self._agent_name} '
            + f'for {self._timescale}'
        ),
        'State': self._state,
        'Chain of thought': prompt.view().text().splitlines(),
    }
    self._history.append(update_log)
