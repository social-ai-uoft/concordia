# Copyright 2024 DeepMind Technologies Limited.
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

"""Ollama Language Model."""

from collections.abc import Collection, Sequence
import re

from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib
from langchain.llms.ollama import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing_extensions import override

MAX_MULTIPLE_CHOICE_ATTEMPTS = 5
MAX_SAMPLE_TEXT_ATTEMPTS = 5

SYSTEM_MESSAGE = (
  'Continue the user\'s sentences. Never repeat their starts. For example, when you see '
  '\'Bob is\', you should continue the sentence after the word \'is\'.'
)

def _extract_choices(text):
  match = re.search(r"\(?(\w)\)", text)
  if match:
    return match.group(1)
  return None


class OllamaLanguageModel(language_model.LanguageModel):
  """Language Model that uses Ollama LLM models."""

  def __init__(
      self,
      model_name: str,
      *,
      system_message: str = "",
      measurements: measurements_lib.Measurements | None = None,
      channel: str = language_model.DEFAULT_STATS_CHANNEL,
      streaming: bool = False,
      **kwargs
  ) -> None:
    """Initializes the instance.

    Args:
        model_name: The language model to use. For more details, see
          https://github.com/ollama/ollama.
        system_message: System message to prefix to requests when prompting the
          model.
        measurements: The measurements object to log usage statistics to.
        channel: The channel to write the statistics to.
    """
    self._model_name = model_name
    self._system_message = system_message
    self._measurements = measurements
    self._channel = channel
    self._terminators = kwargs['terminators'] if 'terminators' in kwargs else []
    if "llama3" in self._model_name and len(self._terminators) == 0:
      self._terminators.extend(['<|eot_id|>'])
    if streaming:
      self._client = Ollama(
        model=model_name,
        stop = self._terminators,
        system=SYSTEM_MESSAGE,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
      )
    else:
      self._client = Ollama(model=model_name, stop=self._terminators)

  @override
  def sample_text(
      self,
      prompt: str,
      *,
      max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
      max_characters: int = language_model.DEFAULT_MAX_CHARACTERS,
      terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
      temperature: float = language_model.DEFAULT_TEMPERATURE,
      timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
      seed: int | None = None,
  ) -> str:
    prompt_with_system_message = f"{self._system_message}\n\n{prompt}"

    terminators = self._terminators.extend(terminators) if terminators is not None else self._terminators

    response = self._client(
        prompt_with_system_message,
        stop=terminators,
        temperature=temperature,
    )

    if self._measurements is not None:
      self._measurements.publish_datum(
          self._channel, {"raw_text_length": len(response)}
      )
    return response

  @override
  def sample_choice(
      self,
      prompt: str,
      responses: Sequence[str],
      *,
      seed: int | None = None,
  ) -> tuple[int, str, dict[str, float]]:
    max_characters = len(max(responses, key=len))
    prompt_with_system_message = f"{self._system_message}\n\n{prompt}"
    sample = self.sample_text(
        prompt_with_system_message,
        max_characters=max_characters,
        temperature=0.0,
        seed=seed,
    )
    answer = _extract_choices(sample)
    try:
      idx = responses.index(answer)
    except ValueError:
      raise language_model.InvalidResponseError(
          f"Invalid response: {answer}. "
          f"LLM Input: {prompt}\nLLM Output: {sample}"
      ) from None

    if self._measurements is not None:
      self._measurements.publish_datum(self._channel, {"choices_calls": 1})
    debug = {}
    return idx, responses[idx], debug

