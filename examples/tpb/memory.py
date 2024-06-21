# Author: Rebekah Gelpi
# Concordia Cognitive Model: Theory of Planned Behaviour

"""
Memory structures for the Theory of Planned Behaviour.
"""
import dataclasses
import datetime

from typing import Callable

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component

from examples.tpb import agent as tpb_agent

@dataclasses.dataclass(kw_only=True)
class WorkingMemory:
  """Simple working memory for storing SARSA-style events"""
  obs_1: str = None
  delib: str = None
  action: str = None
  obs_2: str = None
  reflection: str = None

  def clear(self):
    """Clears the working memory."""
    self.obs_1 = None
    self.delib = None
    self.action = None
    self.obs_2 = None
    self.reflection = None

  def set_any(self, obs_1: str | None=None, delib: str | None=None, action: str | None=None, obs_2: str | None=None, reflection: str | None = None):
    """Sets a value in the working memory."""
    self.obs_1 = obs_1 if obs_1 else self.obs_1
    self.delib = delib if delib else self.delib
    self.action = action if action else self.action
    self.obs_2 = obs_2 if obs_2 else self.obs_2
    self.reflection = reflection if reflection else self.reflection

  def set(self, attribute, value):
    """Sets a value in the working memory."""
    setattr(self, attribute, value)

  def __str__(self):
    """Returns an observation-ready string representation of the working memory"""
    return (
      f"Initial state: {self.obs_1}\n"
      f"Deliberation: {self.delib}\n"
      f"Action: {self.action}\n"
      f"Outcome: {self.obs_2}\n"
      # f"Reflection: {self.reflection} #TODO: Reflection is not yet implemented.
    )

  def next(self):
    """Moves obs_2 to obs_1 and clears the other entries."""
    obs_1 = self.obs_2
    self.clear()
    self.set("obs_1", obs_1)

class EpisodicMemory(component.Component):
  """A memory that stores episodic memories."""
  def __init__(
      self,
      model: language_model.LanguageModel,
      config: tpb_agent.AgentConfig,
      clock_now: Callable[[], datetime.datetime],
      num_memories: int = 100,
      verbose: bool = False,
      **kwargs
    ):
    """
    Initializes the episodic memory.

    Args:
      model: A LanguageModel.
      config: A TPBAgentConfig.
      clock_now: A callable that returns the current time.
      num_memories: The number of memories to retrieve from the memory.
      verbose: Whether to print out the retrieved memories.
    """
    self._model = model
    self._config = config
    self._memory = config.memory
    self._agent_name = config.name
    self._clock_now = clock_now
    self._num_memories = num_memories
    self._verbose = verbose
    self._wm = WorkingMemory()

  def name(self):
    return 'memory'

  def observe(self, observation: str, wm_loc = "obs_1") -> None:
    """Stores the observation in the episodic memory."""

    # Set working memory value.
    self._wm.set(wm_loc, observation.strip())

    if wm_loc == 'obs_2':
      # If the working memory has a full SARSA representation, store the full set of observations...
      if None not in [getattr(self._wm, loc) for loc in ['obs_1', 'delib', 'action', 'obs_2']]:
        ltm_memory = str(self._wm)
      # Otherwise, just log the single observation
      else:
        ltm_memory = observation.strip()

      self._memory.add(f"[observation] {ltm_memory}", timestamp=self._clock_now(), tags=['observation'], importance = 1.)
      # Shift the most recent observation to be the initial state for the next one.
      self._wm.next()

  def summarize(self, observations: str, kind: str = "deliberations") -> str:
    """Summarize the agent's internal deliberations."""

    prompt = interactive_document.InteractiveDocument(self._model)

    prompt.statement(
        f"{kind.capitalize()} of {self._agent_name}: {observations}"
    )

    if kind == "deliberations":
      question = (
        f"Given the above, write a one-sentence summary of the behaviours and their most "
        f"relevant potential consequences for {self._agent_name} and other people that "
        f"{self._agent_name} considered taking when {self._config.pronoun()} was "
        f"making {self._config.pronoun(case = 'genitive')} decision. "
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
