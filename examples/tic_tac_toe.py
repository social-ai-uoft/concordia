import collections
import concurrent.futures
import datetime
import os

from concordia import components as generic_components
from concordia.agents import basic_agent
from concordia.components import agent as components
from concordia.agents import basic_agent
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import embedder_st5
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import game_master as gm_components
from concordia.environment import game_master
from concordia.metrics import goal_achievement
from concordia.metrics import common_sense_morality
from concordia.metrics import opinion_of_others
from concordia.utils import measurements as measurements_lib
from concordia.language_model import gpt_model
from concordia.language_model import gcloud_model
from concordia.utils import html as html_lib
from concordia.utils import plotting

import logging
logging.basicConfig(level=logging.INFO, filename='tic_tac_toe_llama.log')
logger = logging.getLogger('ollama')

embedder = embedder_st5.EmbedderST5()

from concordia.language_model import ollama_model
model = ollama_model.OllamaLanguageModel(
    model_name='llama2:13b', 
    streaming = True
)

#@title Make the clock
time_step = datetime.timedelta(seconds=30)
SETUP_TIME = datetime.datetime(hour=20, year=2024, month=10, day=1)

START_TIME = datetime.datetime(hour=18, year=2024, month=10, day=2)
clock = game_clock.MultiIntervalClock(
    start=SETUP_TIME,
    step_sizes=[time_step, datetime.timedelta(seconds=10)])

#@title Importance models
importance_model = importance_function.ConstantImportanceModel()
importance_model_gm = importance_function.ConstantImportanceModel()

blank_memory_factory = blank_memories.MemoryFactory(
    model=model,
    embedder=embedder,
    importance=importance_model.importance,
    clock_now=clock.now,
)

# This is where a model is prompted to write a backstory - we can customize this
# to use a different prompt or perhaps to load memories from a file.
formative_memory_factory = formative_memories.FormativeMemoryFactory(
    model=model,
    blank_memory_factory_call=blank_memory_factory.make_blank_memory,
)

NUM_PLAYERS = 4

scenario_premise = ["This is a game of tic-tac-toe between two friends."]

player_configs = [
    formative_memories.AgentConfig(
        name='Alice',
        gender='female',
        goal='Wants to win the game.',
        traits='wants to play as X',
    ),
    formative_memories.AgentConfig(
        name='Bob',
        gender='male',
        goal='Wants to win the game.',
        traits='wants to play as O',
    ),
]

def build_agent(
    agent_config,
    player_names: list[str],
    measurements: measurements_lib.Measurements | None = None,
):
  
  mem = blank_memory_factory.make_blank_memory()

  tic_tac_toe_component = components.tic_tac_toe_component.TicTacToe(
    name='tic tac toe component',
      model=model,
      memory=mem,
      agent_name=agent_config.name,
      clock_now=clock.now,
      verbose=True,
  )

  game_policy = components.sequential.Sequential(
      name='game_policy',
      components=[
          tic_tac_toe_component
      ],
  )
  
  current_time_component = components.report_function.ReportFunction(
      name='current_time', function=clock.current_time_interval_str
  )

  current_obs = components.observation.Observation(agent_config.name, mem)
  summary_obs = components.observation.ObservationSummary(
      model=model,
      agent_name=agent_config.name,
      components=[game_policy],
  )

  agent = basic_agent.BasicAgent(
      model,
      mem,
      agent_name=agent_config.name,
      clock=clock,
      verbose=False,
      components=[
          current_obs,
          game_policy,
      ],
      update_interval=time_step,
  )

  return agent

player_configs = player_configs[:NUM_PLAYERS]
player_names = [player.name for player in player_configs][:NUM_PLAYERS]
measurements = measurements_lib.Measurements()

players = []

with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_PLAYERS) as pool:
  for agent in pool.map(build_agent,
                        player_configs[:NUM_PLAYERS],
                        # All players get the same `player_names`.
                        [player_names] * NUM_PLAYERS,
                        # All players get the same `measurements` object.
                        [measurements] * NUM_PLAYERS):
    players.append(agent)


game_master_instructions = """
You are the game master for a tic-tac-toe game between two players.
You are not playing the game yourself.
You are responsible for keeping track of the game state, and for adjudicating the game.
You need to ensure that the game is played fairly, and that the rules are followed.
If a player tries to make an illegal move, you should tell them that they cannot do that.
When the game begins, all cells are empty. In this case, picking any cell is a legal move.
When the game is over, you should announce the winner and end the conversation.
Rules:
1. The game is played on a 3x3 grid.
2. Players take turns placing their mark on an empty cell in the grid.
3. The first player to get three of their marks in a row (horizontally, vertically, or diagonally) wins.
4. If all cells are filled and neither player has won, the game is a draw.
Board Format:
```
  1 | 2 | 3
  ---------
  4 | 5 | 6
  ---------
  7 | 8 | 9
```
"""

game_master_memory = associative_memory.AssociativeMemory(
   sentence_embedder=embedder,
   importance=importance_model_gm.importance,
   clock=clock.now)

# @title Create components of the Game Master
player_names = [player.name for player in players]

instructions_construct = generic_components.constant.ConstantComponent(
    state=game_master_instructions,
    name='Instructions')
scenario_knowledge = generic_components.constant.ConstantComponent(
    state='The game has begun.',
    name='Scenario')

# player_status = gm_components.player_status.PlayerStatus(
#     clock_now=clock.now,
#     model=model,
#     memory=game_master_memory,
#     player_names=player_names)

# convo_externality = gm_components.conversation.Conversation(
#     players=players,
#     model=model,
#     memory=game_master_memory,
#     clock=clock,
#     burner_memory_factory=blank_memory_factory,
#     components=[player_status],
#     cap_nonplayer_characters=3,
#     game_master_instructions=game_master_instructions,
#     shared_context=shared_context,
#     verbose=False,
# )

# direct_effect_externality = gm_components.direct_effect.DirectEffect(
#     players=players,
#     model=model,
#     memory=game_master_memory,
#     clock_now=clock.now,
#     verbose=False,
#     components=[player_status]
# )

tic_tac_toe_gm = gm_components.tic_tac_toe_gm.TicTacToeGame(
    model=model,
    player_names=player_names,
    verbose=False,
)

# @title Create the game master object
env = game_master.GameMaster(
    model=model,
    memory=game_master_memory,
    clock=clock,
    players=players,
    components=[
        instructions_construct,
        scenario_knowledge,
        tic_tac_toe_gm
    ],
    randomise_initiative=False,
    player_observes_event=True,
    verbose=True,
)

clock.set(START_TIME)

for premis in scenario_premise:
  game_master_memory.add(premis)
  for player in players:
    player.observe(premis)


episode_length = 10  # @param {type: 'integer'}
for _ in range(episode_length):
  env.step()

