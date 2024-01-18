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


"""This is a factory for generating memories for generative agents."""

from collections.abc import Callable
import datetime

from concordia.associative_memory import associative_memory
from concordia.language_model import language_model
import numpy as np


class MemoryFactory:
  """Generator of formative memories."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      embedder: Callable[[str], np.ndarray],
      importance: Callable[[str], float],
      clock_now: Callable[[], datetime.datetime],
  ):
    self._model = model
    self._embedder = embedder
    self._importance = importance
    self._clock_now = clock_now

  def make_blank_memory(
      self,
  ) -> associative_memory.AssociativeMemory:
    """Creates a blank memory.

    Returns a blank memory

    Returns:
      An empty memory structure
    """

    return associative_memory.AssociativeMemory(
        self._embedder,
        self._importance,
        clock=self._clock_now,
    )
