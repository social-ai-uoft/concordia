"""Agent components for planning."""

from collections.abc import Sequence
import datetime
from typing import Callable
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component

from examples.tpb.components import TPBComponent
from examples.tpb import agent as tpb_agent

class TPBPlan(TPBComponent):
  """Theory of Planned Behaviour plan component."""
  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      config: tpb_agent.AgentConfig,
      components: Sequence[component.Component] | None = None,
      num_memories: int = None,
      goal: component.Component | None = None,
      timescale: str = 'the rest of the day',
      time_adverb: str = 'hourly',
      clock_now: Callable[[], datetime.datetime] | None = None,
      verbose: bool = False,
      **kwargs
  ):
    """Initialize the Plan component.
    
    Args:
      name: The name of the component.
      model: The language model.
      memory: The agent memory.
      clock_now: Time callback function.
      verbose: Whether to print the state.
    """

    super().__init__(name=name,model=model,config=config,components=components,num_memories=num_memories,verbose=verbose,clock_now=clock_now,**kwargs)
    self._goal = goal
    self._timescale = timescale
    self._time_adverb = time_adverb
    self._is_initialized = True
    self._current_plan = ""

  def _update(self):

    goal_state = self._goal.state() if self._goal is not None else ""
    print(self._goal)
    print(goal_state)
    memories = "\n".join(
      self._memory.retrieve_associative(
        goal_state,
        self._num_memories,
        self._timescale
      )
    )

    context = "\n".join(
      [component.state() for component in self._components]
    )

    prompt = interactive_document.InteractiveDocument(model=self._model)

    statements = {
        "Memories": memories,
        "Goal": goal_state,
        "Situation": context,
        "Current time": self._clock_now()
      }
    
    if self._current_plan:
      statements["Current plan"] = self._current_plan

    self.prompt_builder(
      prompt,
      statements
    )

    if self._current_plan:
      should_replan = prompt.yes_no_question(
        f"Given the above, should {self._agent_name} change {self._config.pronoun(case = 'genitive')} current plan?"
      )
    else:
      should_replan = True

    if should_replan or not self._state:
      self._current_plan = prompt.open_question(
        f"Write {self._agent_name}'s plan for {self._timescale}. Please, "
        f"provide a {self._time_adverb} schedule, keeping in mind "
        f"{self._agent_name}'s situation and goal.\n"
        f"Please format the plan like in this example: [21:00 - 22:00] watch TV"
      )

      self._state = self._current_plan
      self._last_chain = prompt
