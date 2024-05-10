import concurrent.futures

from typing import Callable, Sequence

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains
from concordia.typing import game_master, clock, agent, component

DEFAULT_THOUGHTS = (
  thought_chains.attempt_to_result,
  thought_chains.result_to_causal_statement
)


class SceneEnvironment(game_master.GameMaster):

  """Custom scene environment/'game master' class for the TPB agent."""

  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      players: Sequence[agent.GenerativeAgent],
      clock: clock.GameClock,
      components: Sequence[component.Component],
      update_thought_chain: Sequence[
        Callable[[interactive_document.InteractiveDocument, str, str], str]
      ] = DEFAULT_THOUGHTS,
      **kwargs
  ):

    self._model = model
    self._memory = memory
    self._players = {}
    for player in players:
      self._players[player.name()] = player
    self._clock = clock
    self._components = {}
    for component in components:
      self._components[component.name()] = component
    self._update_from_player_thoughts: update_thought_chain
    self._kwargs = kwargs

  def name(self) -> str:
    return "Scene Environment"

  def get_memory(self) -> associative_memory.AssociativeMemory:
    return self._memory

  def update_from_player(self, player_name: str, action_attempt: str) -> str:

    """
    Given an action attempt from a player, use a thought chain to determine the
    outcome of an event and let the player observe the results.

    Args:
      player_name: The name of the player involved.
      action_attempt: The attempted behaviour of the player.
    """

    prompt = interactive_document.InteractiveDocument(model=self._model)

    # Provide single-threading for pre-event component updates
    # (I don't think we will need this, but it is here for the moment anyway)
    if self._kwargs['threading']:
      with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.map(
            lambda construct: construct.update_before_event(
                f'{player_name}: {action_attempt}'
            ),
            self._components.values(),
        )
    else:
      for component in self._components.values():
        component.update_before_event(f'{player_name}: {action_attempt}')

    # Produce the event that has happened as the result of the action attempt
    prompt, event_statement = thought_chains.run_chain_of_thought(
        self._update_from_player_thoughts,
        action_attempt,
        prompt,
        player_name,
    )

    # Game master observes
    self._memory.add(event_statement)

    # Player observes event
    self._players[player_name].observe(event_statement)

  def view_for_player(self, player_name: str) -> str:
    pass

  def step(
      self,
      *,
      active_players: Sequence[agent.GenerativeAgent] | None = None,
      action_spec: agent.ActionSpec | None = None,
      ):
    pass



  def run_episode(self, max_steps: int) -> Sequence[str]:
    for _ in range(max_steps):
      self.step()
