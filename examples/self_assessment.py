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
logging.basicConfig(level=logging.INFO, filename='self_assessment_ollama.log')
logger = logging.getLogger('ollama')

embedder = embedder_st5.EmbedderST5()

from concordia.language_model import ollama_model
model = ollama_model.OllamaLanguageModel(
    model_name='llama2:13b', 
    # streaming = True
)

#@title Make the clock
time_step = datetime.timedelta(minutes=20)
SETUP_TIME = datetime.datetime(hour=20, year=2024, month=10, day=1)

START_TIME = datetime.datetime(hour=18, year=2024, month=10, day=2)
clock = game_clock.MultiIntervalClock(
    start=SETUP_TIME,
    step_sizes=[time_step, datetime.timedelta(seconds=10)])

#@title Importance models
importance_model = importance_function.ConstantImportanceModel()
importance_model_gm = importance_function.ConstantImportanceModel()

shared_memories = [
    "You are enrolled in a computer science program at your local university.",
    "So far, you've learned about the basics of programming, data structures, and algorithms.",
    "You're trying to decide which courses to take next semester.",
    "You're also trying to decide whether you're read to apply for an internship in machine learning and artificial intelligence.",
]

shared_context = model.sample_text(
    'Summarize the following passage in a concise and insightful fashion:\n'
    + '\n'.join(shared_memories)
    + '\n'
    + 'Summary:'
)
print(shared_context)

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
    shared_memories=shared_memories,
    blank_memory_factory_call=blank_memory_factory.make_blank_memory,
)

NUM_PLAYERS = 4

scenario_premise = [
  "Alice is a first-year computer science student at a local university.",
  "She is trying to decide which courses to take next semester.",
]

player_configs = [
    formative_memories.AgentConfig(
        name='Alice',
        gender='female',
        goal='Alice wants to have a successful career and a balanced, happy life.',
        context=shared_context,
        traits='{curious, hard-working, ambitious, intelligent}',
    ),
]

def build_agent(
    agent_config,
    player_names: list[str],
    measurements: measurements_lib.Measurements | None = None,
):
  mem = formative_memory_factory.make_memories(agent_config)

#   self_perception = components.self_perception.SelfPerception(
#       name='self perception',
#       model=model,
#       memory=mem,
#       agent_name=agent_config.name,
#       clock_now=clock.now,
#       verbose=True,
#   )
#   situation_perception = components.situation_perception.SituationPerception(
#       name='situation perception',
#       model=model,
#       memory=mem,
#       agent_name=agent_config.name,
#       clock_now=clock.now,
#       verbose=True,
#   )
#   person_by_situation = components.person_by_situation.PersonBySituation(
#       name='person by situation',
#       model=model,
#       memory=mem,
#       agent_name=agent_config.name,
#       clock_now=clock.now,
#       components=[self_perception, situation_perception],
#       verbose=True,
#   )

  mood_by_situation = components.mood_by_situation.MoodBySituation(
    name='mood by situation',
      model=model,
      memory=mem,
      agent_name=agent_config.name,
      clock_now=clock.now,
      verbose=True,
  )

  persona = components.sequential.Sequential(
      name='persona',
      components=[
        #   self_perception,
        #   situation_perception,
        #   person_by_situation,
          mood_by_situation
      ],
  )
  
  current_time_component = components.report_function.ReportFunction(
      name='current_time', function=clock.current_time_interval_str
  )

  current_obs = components.observation.Observation(agent_config.name, mem)
  summary_obs = components.observation.ObservationSummary(
      model=model,
      agent_name=agent_config.name,
      components=[persona],
  )

  agent = basic_agent.BasicAgent(
      model,
      mem,
      agent_name=agent_config.name,
      clock=clock,
      verbose=False,
      components=[
          persona,
          current_time_component,
          summary_obs,
          current_obs,
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


game_master_instructions = (
    'You are a life coach. Your job is to help your clients achieve their goals.'
    'You are responsible for suggesting actions that your clients can take to achieve their goals.'
    'You are also resonsible for evaluating the actions that your clients take.'
    'You are also responsible for maintaining records related to your clients\' goals and actions.'
)

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
    state=' '.join(shared_memories),
    name='Background')

player_status = gm_components.player_status.PlayerStatus(
    clock_now=clock.now,
    model=model,
    memory=game_master_memory,
    player_names=player_names)

convo_externality = gm_components.conversation.Conversation(
    players=players,
    model=model,
    memory=game_master_memory,
    clock=clock,
    burner_memory_factory=blank_memory_factory,
    components=[player_status],
    cap_nonplayer_characters=3,
    game_master_instructions=game_master_instructions,
    shared_context=shared_context,
    verbose=False,
)

direct_effect_externality = gm_components.direct_effect.DirectEffect(
    players=players,
    model=model,
    memory=game_master_memory,
    clock_now=clock.now,
    verbose=False,
    components=[player_status]
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
        player_status,
        convo_externality,
        direct_effect_externality,
    ],
    randomise_initiative=True,
    player_observes_event=False,
    verbose=True,
)

clock.set(START_TIME)

for premis in scenario_premise:
  game_master_memory.add(premis)
  for player in players:
    player.observe(premis)


episode_length = 3  # @param {type: 'integer'}
for _ in range(episode_length):
  env.step()

