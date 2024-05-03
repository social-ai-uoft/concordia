# Author: Rebekah Gelpi
# Concordia Cognitive Model: Theory of Planned Behaviour

"""Agent component for self perception."""
import datetime
import re
import concurrent.futures
import json
import os
import termcolor
import numpy as np

from typing import Callable, Sequence, Union

from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component

from examples.custom_components import utils
from examples.tpb import plan as plan


class BasicEpisodicMemory(component.Component):
  """Basic component that synthesizes observations"""
  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      clock_now: Callable[[], datetime.datetime] | None = None,
      timeframe: datetime.timedelta = datetime.timedelta(hours=1),
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
  
    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._state = ''
    self._config = player_config
    self._agent_name = player_config.name
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = "memory"
    self._timeframe = timeframe

    # A dictionary inspired by SARSA containing the:
    # - Observation of preceding event by agent
    # - Deliberation/Cognition of agent
    # - Action of agent
    # - Observation of consequences/resulting event.
    self._working_memory = {
      "O": "",
      "D": "",
      "A": "",
      "O2": "",
      "R": ""
    }

  def name(self) -> str:
    return self._name

  def observe(self, observation: str, wm_loc: str = "O2") -> None:
    """Take an observation and add it to the memory."""
    
    # Basic tag
    tags = ['observation']

    self._working_memory[wm_loc] = observation.strip()

    # If GM submits something, it is O2.
    # GM submits observations in the direct effect module.
    # We submit observations in the SequentialTPBModel: 
    # D after deliberation
    # A after action
    # O2 is moved to O after we get a new O2.

    # If there is a hyphen in the observation, add the "conversation tag."
    if " -- " in observation:
      tags.append('conversation')

    if wm_loc == "O2":
      # If the initial state, action, and plan are filled out, then build a SARSA memory:
      if self._working_memory['O'] and self._working_memory['D'] and self._working_memory['A']:
        ltm_memory = (
          f"Initial state: {self._working_memory['O']}\n"
          f"Deliberation: {self._working_memory['D']}\n"
          f"Plan: {self._working_memory['A']}\n"
          f'Consequences: {self._working_memory["O2"]}\n'
          # f'Reflections: {self._working_memory["R"]}\n' # TODO: Reflection?
        )
      # Otherwise, just log a simple observation.
      else:
        ltm_memory = observation.strip()
      # Add the working memory to the LTM
      importance = 1.
      self._memory.add(f'[{", ".join(tags)}] {ltm_memory}', timestamp=self._clock_now(), tags=tags, importance=importance)
      # Move the resulting observation into the initial observation for the next state
      self._working_memory = {"O": f'{self._working_memory["O2"]}',"D": "","A": "","O2": ""}


  def summarize(self, observations: list[str], kind: str = "deliberations") -> str:
    """Summarize the agent's internal deliberations."""

    prompt = interactive_document.InteractiveDocument(self._model)
    
    observations = observations

    prompt.statement(
        f"{kind.capitalize()} of {self._agent_name}: {observations}"
    )

    if kind == "deliberations":
      question = (
        f"Given the above, write a one-sentence summary of the behaviours and their most "
        f"relevant potential consequences for {self._agent_name} and other people that "
        f"{self._agent_name} considered taking when {utils.pronoun(self._config)} was "
        f"making {utils.pronoun(self._config, case = 'genitive')} decision. "
      )
    elif kind == "plan":
      question = (
        f"Restate {self._agent_name}'s plan in a single sentence. If the {kind} includes "
        f"dialogue, make sure to include the name of the person who is talking."
      )
    else:
      question = (
        f"Restate the {kind} in a single sentence. If the {kind} includes dialogue, make "
        f"sure to include the name of the person who is talking."
      )

    summary = prompt.open_question(
        question,
        max_tokens=1000,
        max_characters=1200,
        terminators=()
    )

    return summary

class DialogueMemory(component.Component):
  """Component specialized for storing and displaying dialogue."""
  def __init__(
      self,
      model: language_model.LanguageModel,
      player_config: formative_memories.AgentConfig,
      memory: associative_memory.AssociativeMemory,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 10,
      verbose: bool = False
  ):
    
    self._name = "dialogue memory"
    self._model = model
    self._player_config = player_config
    self._memory = memory
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._verbose = verbose

  def name(self) -> str:
    return self._name

  def is_dialogue(self, observation: str) -> bool:
    """Check whether the given observation is a dialogue observation or not."""
    prompt = interactive_document.InteractiveDocument(model=self._model)
    question = (
      f'Is someone speaking?\n'
      f'{observation}\n'
    )
    return prompt.yes_no_question(question=question)

  def is_dialogue_over(self) -> bool:
     """Check whether a conversation has finished."""
     conversation = "\n".join(self._memory.retrieve_recent(self._num_memories_to_retrieve))
     prompt = interactive_document.InteractiveDocument(model=self._model)
     question = (
       f'Has the following conversation ended?\n'
       f'{conversation}'
     )
     return prompt.yes_no_question(question=question)

  def in_conversation(self) -> bool:
    """Check whether the player is currently in a conversation."""
    observation = "\n".join(self._memory.retrieve_recent(1))
    prompt = interactive_document.InteractiveDocument(model=self._model)
    question = (
      f'Is someone speaking?\n'
      f'{observation}\n'
    )
    return prompt.yes_no_question(question=question)

  def retrieve_conversation(self) -> list[str]:
    """Retrieve a conversation from the dialogue memory."""
    return self._memory.retrieve_recent(self._num_memories_to_retrieve) if len(self._memory) > 0 else []

  def observe(self, observation: str) -> None:
    """Store the given observation in the dialogue memory."""
    if self.is_dialogue(observation):
      self._memory.add(observation)