from collections.abc import Callable, Sequence
import concurrent
import contextlib
import copy
import datetime
import threading
import termcolor

from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import agent
from concordia.typing import clock as game_clock
from concordia.typing import component
from concordia.utils import helper_functions

from examples.tpb import tpb
from examples.tpb import memory as tpb_memory


class TPBAgent(basic_agent.BasicAgent):
  """
  Extension of the basic agent specialized for using the TPB component architecture.
  """

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      clock: game_clock.GameClock,
      dialogue_memory: tpb_memory.DialogueMemory,
      components: Sequence[component.Component] | None = None,
      num_memories_retrieved: int = 10,
      update_interval: datetime.timedelta = datetime.timedelta(hours=1),
      verbose: bool = False,
      user_controlled: bool = False,
      print_colour='green',
  ):
    """A generative agent.

    Args:
      model: a language model
      memory: an associative memory
      agent_name: the name of the agent
      clock: the game clock is needed to know when is the current time
      components: components that contextualise the policies. The components
        state will be added to the agents state in the order they are passed
        here.
      num_memories_retrieved: number of memories to retrieve for acting,
        speaking, testing
      update_interval: how often to update components. In game time according to
        the clock argument.
      verbose: whether to print chains of thought or not
      user_controlled: if True, would query user input for speech and action
      print_colour: which colour to use for printing
    """

    super().__init__(model=model, memory=memory, agent_name=agent_name,clock=clock,components=components,
                     num_memories_retrieved=num_memories_retrieved,update_interval=update_interval,
                     verbose=verbose, user_controlled=user_controlled, print_colour=print_colour)
    
    # TPB specific
    self._in_conversation = False
    self._dialogue_memory = dialogue_memory

  def observe(self, observation: str) -> None:
    
    if observation and not self._under_interrogation:
      for comp in self._components.values():
        comp.observe(observation)

      if self._dialogue_memory.is_dialogue(observation):
        self._in_conversation = True
        self._dialogue_memory.observe(observation)
      if self._dialogue_memory.is_dialogue_over():
        self._in_conversation = False

      
    


