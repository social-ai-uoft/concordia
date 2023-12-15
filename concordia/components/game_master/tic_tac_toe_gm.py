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


"""This construct track the status and location of players."""

from collections.abc import Callable, Sequence
from typing import Tuple
import datetime

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component


class TicTacToeGame(component.Component):
  """Tracks the state of a tic-tac-toe game."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      player_names: Sequence[str],
      verbose: bool = False,
  ):
    self._model = model
    self._state = [[' ']*3 for _ in range(3)]  # 3x3 empty tic-tac-toe board
    self._player_names = player_names
    self._current_player_index = 0  # Index of the current player in player_names
    self._verbose = verbose
    self._history = []

  def name(self) -> str:
    return 'TicTacToeGame'

  def state(self) -> str:
    return '\n' + '\n'.join(['   |   |   ' if all(cell == ' ' for cell in row) else ' | '.join(row) for row in self._state])

  def get_history(self):
    return self._history.copy()

  def get_last_log(self):
    if self._history:
      return self._history[-1].copy()

  def update(self, move: Tuple[int, int]) -> None:
    """Update the game state with a move.

    Args:
      move: A tuple of two integers representing the row and column where the current player wants to place their mark.
    """
    row, col = move
    if self._state[row][col] != ' ':
      raise ValueError('Invalid move. The cell is already occupied.')
    self._state[row][col] = 'X' if self._current_player_index == 0 else 'O'
    self._current_player_index = (self._current_player_index + 1) % len(self._player_names)

    if self._verbose:
      print(self.state())

    self._history.append({
        'date': datetime.datetime.now(),
        'state': self.state(),
        'move': move,
        'player': self._player_names[self._current_player_index]
    })