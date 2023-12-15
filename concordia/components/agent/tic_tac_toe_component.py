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

"""Agent component for situation perception."""
import datetime
from typing import Callable

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class TicTacToe(component.Component):
  """This component plays a game of tic-tac-toe."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 25,
      verbose: bool = False,
  ):
    """Initializes the component.

    Args:
      name: The name of the component.
      model: The language model to use.
      memory: The memory to use.
      agent_name: The name of the agent.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: The number of memories to retrieve.
      verbose: Whether to print the last chain.
    """
    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = name

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  def update(self) -> None:
    prompt = interactive_document.InteractiveDocument(self._model)

    memories = self._memory.retrieve_recent(
        self._num_memories_to_retrieve, add_time=True)

    prompt.statement(f'Recent game state:\n{memories}')

    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')

    question = (
        f'Given the current board state, where should {self._agent_name} place their mark? You answer should be a precisely position number between 1 and 9.'
    )

    self._state = prompt.open_question(
        question,
        answer_prefix='',
        max_characters=3000,
        max_tokens=1000,
    )

    if len(self._state) == 0:
      raise ValueError('TicTacToe component returned empty state.')

    self._last_chain = prompt
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')
