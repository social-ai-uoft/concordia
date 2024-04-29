import datetime
from sentence_transformers import SentenceTransformer

from concordia.agents import basic_agent
from concordia.components import agent as components
from concordia import components as generic_components
from concordia.associative_memory import associative_memory
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.components import game_master as gm_components
from concordia.environment import game_master
from concordia.language_model import language_model
from concordia.metrics import goal_achievement
from concordia.metrics import common_sense_morality
from concordia.metrics import opinion_of_others
from concordia.utils import html as html_lib
from concordia.utils import measurements as measurements_lib
from concordia.utils import plotting

# decorator for model
import functools
def default_kwargs(**defaultKwargs):
  def actual_decorator(fn):
    @functools.wraps(fn)
    def g(*args, **kwargs):
      defaultKwargs.update(kwargs)
      return fn(*args, **defaultKwargs)
    return g
  return actual_decorator

@default_kwargs(USE_CLOUD = False, streaming=True, transformer_name = 'Alibaba-NLP/gte-large-en-v1.5')
def model_setup(
    model_name: str,
    local_models: bool = True,
    **kwargs
) -> tuple[language_model.LanguageModel, SentenceTransformer.encode]:
  """Return a tuple of relevant model objects
  
  Args:
    model_name: The name of the model to use.
    local_models: Whether to use local models or API. (Default: True)"""
  if local_models:
    # Setup language model
    from concordia.language_model import ollama_model
    model = ollama_model.OllamaLanguageModel(
      model_name='llama3:70b',
      streaming=kwargs['streaming']
    )

  else:
    # Use API access
    assert ("CLOUD_PROJECT_ID" or "GPT_API_KEY") in kwargs, "Need to provide a GCloud ID or GPT API key."
    from concordia.language_model import gpt_model
    from concordia.language_model import gcloud_model
    if kwargs["USE_CLOUD"]:
      model = gcloud_model.CloudLanguageModel(project_id=kwargs["CLOUD_PROJECT_ID"])
    else:
      model = gpt_model.GptLanguageModel(api_key=kwargs["GPT_API_KEY"], model_name=model_name)
  
  # Setup sentence encoder
  st5_model = SentenceTransformer(kwargs['transformer_name'],trust_remote_code=True) 
  embedder = st5_model.encode

  return model, embedder

def measurement_setup(
    SETUP_TIME: datetime.date = datetime.datetime(hour=20, year=2024, month=10, day=1),
    START_TIME: datetime.date = datetime.datetime(hour=18, year=2024, month=10, day=2),
    time_step = datetime.timedelta(minutes=20)
) -> tuple[measurements_lib.Measurements, game_clock.MultiIntervalClock]:
  # Setup measurements and clock
  measurements = measurements_lib.Measurements()

  # Clock setup
  clock = game_clock.MultiIntervalClock(
    start=SETUP_TIME,
    step_sizes=[time_step, datetime.timedelta(seconds=10)])
  
  return measurements, clock