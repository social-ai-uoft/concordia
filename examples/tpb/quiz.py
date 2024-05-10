"""Questionnaire component for the TPB task scene."""

import json
import os

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component

from examples.tpb import agent as tpb_agent

class Quiz(component.Component):
  """Questionnaire component for the TPB task scene."""
  def __init__(
    self,
    model: language_model.LanguageModel,
    config: tpb_agent.AgentConfig,
    questions: str | os.PathLike,
    verbose: bool = False
  ):
    """
    Initializes a Quiz component.

    Args:
      model: Language model.
      config: TPB agent config.
      questions: Either a parsable JSON string containing the questionnaire, or a path to a JSON file containing the questionnaire.
      verbose: Whether to print debug information.
    """

    self._name = "Quiz"
    self._state = ""
    self._model = model
    self._config = config
    self._agent_name = config.name
    self._memory = config.memory
    self._traits = config.traits
    self._verbose = verbose
    self._results = {}

    # Questions must be either a parseable JSON string or a file path.
    if not os.path.exists(questions):
      self._quiz = json.loads(questions)
    else:
      with open(questions, "r") as f:
        self._quiz = json.load(f)

  @property
  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  @property
  def results(self) -> dict:
    return self._results

  def update_results(self, key, value):
    if key not in self._results:
      self._results[key] = value

  def observe(self, observation: str) -> None:

    for question in self._quiz["questions"]:

      prompt = interactive_document.InteractiveDocument(model=self._model)
      nl = "\n"
      prompt.statement(f"Memories of {self._agent_name}: {nl.join(
        self._memory.retrieve_associative(observation, k = 10)
      )}")
      prompt.statement(f"Traits of {self._agent_name}: {self._traits}")
      prompt.statement(f"Observation: {observation}")

      answer = prompt.multiple_choice_question(
        question["question"], question["options"]
      )

      self.update_results(question["question"], question["options"][answer])
