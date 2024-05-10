# Author: Rebekah Gelpi
# Concordia Cognitive Model: Theory of Planned Behaviour Agent

import dataclasses
import datetime
import os
import pandas as pd
import pickle

from typing import Any, Iterable

from concordia.associative_memory import (
  associative_memory, formative_memories,
  blank_memories, importance_function
)
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import agent, component, clock
from concordia.utils import helper_functions

from examples.tpb import utils

DEFAULT_DOB = datetime.datetime(year=1980, month=7, day=23, hour=0, minute=0)
DEFAULT_FORMATIVE_AGES = (6, 9, 13, 16, 21)
DEFAULT_IMPORTANCE_MODEL = importance_function.ConstantImportanceModel()

@dataclasses.dataclass(kw_only=True)
class AgentConfig:
  """
  A card that describes a TPB agent.
  """
  name: str
  gender: str
  context: str = ''
  goal: str = ''
  traits: str = ''
  specific_memories: str = ''
  date_of_birth: datetime.datetime = DEFAULT_DOB
  formative_ages: Iterable[int] = DEFAULT_FORMATIVE_AGES
  formative_memory_importance: float = 1.0
  extras: dict[str, Any] = dataclasses.field(default_factory=dict)
  # model: str
  # embedder: str
  # clock_now = Callable[[], datetime.datetime]
  # importance = Callable[[], float]

  def __post_init__(self):
    """Post-initialization hook."""
    self.pronoun = lambda case="nominative": utils.pronoun(self.gender, case=case)
    self.age = lambda now: f"{utils.format_timedelta(now - self.date_of_birth)} old"

    # If a memory can be built from the dictionary keys, use them to build formative memories.
    mem_params = ['model', 'embedder', 'clock_now', 'importance']
    if all(param in self.extras.keys() for param in mem_params):
      self.memory = None
      memory_factory = blank_memories.MemoryFactory(self.extras['model'], self.extras['embedder'], self.extras['importance'], self.extras['clock_now'])
      formative_memory_factory = formative_memories.FormativeMemoryFactory(
        model=self.extras['model'],
        shared_memories=(self.context,),
        blank_memory_factory_call=memory_factory.make_blank_memory)
      # Use the agent configuration to make formative memories.
      self.memory: associative_memory.AssociativeMemory = formative_memory_factory.make_memories(self)
    # If no memories are provided, a memory can also be passed into the dictionary.
    else:
      if 'memory' in self.extras.keys():
        self.memory: associative_memory.AssociativeMemory = self.extras['memory']
      else:
        self.memory: associative_memory.AssociativeMemory = None

  def add_memory(self, memory: associative_memory.AssociativeMemory):
    """Memory setter."""
    self.memory = memory

  def save_config(self, file_path = str | os.PathLike) -> None:
    """Save the agent configuration."""
    with open(file_path, 'wb') as f:
      pickle.dump({
        'name': self.name,
        'gender': self.gender,
        'context': self.context,
        'goal': self.goal,
        'traits': self.traits,
        'date_of_birth': self.date_of_birth,
        'formative_ages': self.formative_ages,
        'formative_memory_importance': self.formative_memory_importance,
        'extras': {
          'memory': self.memory.get_data_frame() if self.memory else pd.DataFrame(columns=['text', 'timestamp', 'tags', 'importance'])
        }}, f)

  @classmethod
  def load_config(cls, file_path = str | os.PathLike):
    """Load the agent configuration."""
    with open(file_path, 'rb') as f:
      config_dict = pickle.load(f)
    return cls(**config_dict)

class TPBAgent(agent.GenerativeAgent):
  """
  Agent that implements a TPB model.
  """
  def __init__(
      self,
      config: AgentConfig,
      full_model: component.Component,
      clock: clock.GameClock,
  ):

    self._config = config
    self._clock = clock
    self._model = config.extras['model']
    self._full_model = full_model
    self._log = []
    self.update()

  @property
  def name(self) -> str:
    return self._config.name

  def state(self) -> str:
    return self._full_model.state()

  def update(self) -> None:
    self._full_model.update()

  def get_last_log(self):
    if not self._log:
      return ''
    return self._log[-1]

  def act(
      self,
      action_spec: agent.ActionSpec = agent.DEFAULT_ACTION_SPEC,
      memorize: bool = False,
  ):

    if not action_spec:
      action_spec = agent.DEFAULT_ACTION_SPEC
    self.update()
    prompt = interactive_document.InteractiveDocument(self._model)
    context_of_action = '\n'.join([
        f'{self.state()}',
    ])

    prompt.statement(context_of_action)

    call_to_action = action_spec.call_to_action.format(
        agent_name=self.name,
        timedelta=helper_functions.timedelta_to_readable_str(
            self._clock.get_step_size()
        ),
    )
    output = ''

    if action_spec.output_type == 'FREE':
      output = self.name + ' '
      output += prompt.open_question(
          call_to_action,
          max_characters=2500,
          max_tokens=2200,
          answer_prefix=output,
      )
    elif action_spec.output_type == 'CHOICE':
      idx = prompt.multiple_choice_question(
          question=call_to_action, answers=action_spec.options
      )
      output = action_spec.options[idx]
    elif action_spec.output_type == 'FLOAT':
      raise NotImplementedError

    self._last_chain_of_thought = prompt.view().text().splitlines()

    current_log = {
        'date': self._clock.now(),
        'Action prompt': self._last_chain_of_thought,
    }

    self._log.append(current_log)

  def observe(self, observation: str) -> None:

    self._full_model.observe(observation=observation)
