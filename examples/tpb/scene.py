
from collections.abc import Sequence

from concordia.agents.basic_agent import BasicAgent
from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.thought_chains import thought_chains
from concordia.typing.clock import GameClock
from concordia.typing.component import Component
from concordia.typing.game_master import GameMaster




class TPBGameMaster(GameMaster):
  """Custom Game Master for the TPB task."""
  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      clock: GameClock,
      players: Sequence[BasicAgent],
      components: Sequence[Component],

  ):
    self._model = model
    self._memory = memory
    self._clock = clock
    self._players: dict[str, BasicAgent] = {}
    for player in players:
      self._players[player.name] = player
    self._components: dict[str, Component] = {}
    for component in components:
      self._components[component.name()] = component
    
  @property
  def name(self) -> str:
    return "TPB Game Master"
  
  def _update_components(self):
    for component in self._components.values():
      component.update()
  
  def update_from_player(self, player_name: str, action_attempt: str) -> str:

    # Open prompt
    prompt = interactive_document.InteractiveDocument(self._model)

    # Update GM components:
    # - What is the player status?
    # - What is the player's situation?
    # - Constant component: Information about the player? (relevant to TPB)
    self._update_components()

    # Add the component values to the prompt
    for component in self._components.values():
      prompt.statement(component.name() + ': ' + component.state() + '\n')

    # Get the outcome of the event.
    prompt, event_statement = thought_chains.run_chain_of_thought(
        thought_chains.attempt_to_result,
        action_attempt,
        prompt,
        player_name,
    )

    self._memory.add(event_statement)

    self._players[player_name].observe(event_statement, wm_loc = "obs_2")

    

    





    


  def view_for_player(self, player_name: str) -> str:
    pass

  def run_episode(self, max_steps: int) -> Sequence[str]:
    pass