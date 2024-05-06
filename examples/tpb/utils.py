import datetime
import json
import os
import pickle

import numpy as np
from numpy.typing import ArrayLike

from concordia.agents import basic_agent
from concordia.associative_memory import formative_memories, associative_memory, blank_memories
from concordia.typing import component

##################################
# region: Basic helper functions #
##################################

def pronoun(
    gendered: basic_agent.BasicAgent | formative_memories.AgentConfig | str,
    case: str = "nominative") -> str:
  """Gets the pronoun for the specified gender.
  
  By default, uses the `gender` property if the input object (e.g., BasicAgent or AgentConfig) has one;
  otherwise, it will treat the input as the gender itself.
  
  Args:
    gendered: An object with a `gender` property, or a string indicating a gender.
    case: The grammatical case of the word. Options: `['nominative', 'oblique', 'genitive', 'genitive-s']`."""

  if hasattr(gendered, 'name'):
    _gender = gendered.gender
  else:
    _gender = gendered

  she = {
    "nominative": "she",
    "oblique": "her",
    "genitive": "her",
    "genitive-s": "hers"
  }
  he = {
    "nominative": "he",
    "oblique": "him",
    "genitive": "his",
    "genitive-s": "his"
  }
  they = {
    "nominative": "they",
    "oblique": "them",
    "genitive": "their",
    "genitive-s": "theirs"
  }

  return she[case] if _gender.lower() in ["female", "woman", "girl", "f"] else he[case] if _gender.lower() in ["male", "man", "boy", "m"] else they[case]

def format_timedelta(time_delta: datetime.timedelta, unit: str = None) -> str:
  """Format a `datetime.timedelta` object to return a length of time as a string,
  e.g. '2 hours', with the nearest single- or double-digit integer quantity. If the number
  can be represented by 1 or more of the next largest quantity, the unit used
  should be the larger one. For example, `datetime.timedelta(seconds=75)` should
  return '1 minute'.
  
  Args:
    time_delta: A `datetime.timedelta` object. 
    unit: (Optional) a string containing a unit to override."""

  secs = max(time_delta, datetime.timedelta(seconds=1)).total_seconds()

  _second_div = 1
  _minute_div = 60*_second_div
  _hour_div = 60*_minute_div
  _day_div = 24*_hour_div
  _year_div = 365*_day_div

  divs = {
    "seconds": _second_div,
    "minutes": _minute_div,
    "hours": _hour_div,
    "days": _day_div,
    "years": _year_div
  }

  for _unit, _div in divs.items():
    if (secs / _div) < 1:
      break
    denominator = _div
    unit_str = _unit
  
  # Optionally override the unit
  denominator = denominator if unit is None else divs[unit]
  output_val = round(secs / denominator)
  unit_str = unit_str if output_val != 1 else unit_str[:-1]
  
  return f"{output_val} {unit_str}"

def softmax(x: ArrayLike, tau: float):
  """Returns a softmax probability distribution.
  
  Args:
    x: An `np.ndarray`, or `ArrayLike` object that supports conversion into a `np.ndarray`.
    tau: The inverse temperature. Higher inverse temperatures result in more deterministic choices."""
  x = np.asarray(x) if not isinstance(x, np.ndarray) else x
  return np.exp(tau*x) / sum(np.exp(tau*x))

def normalize(x: ArrayLike):
  """Normalizes an array of values."""
  x = np.asarray(x) if not isinstance(x, np.ndarray) else x
  return x / sum(x)

##################################
# endregion                      #
##################################

##################################
# region Loading mems/components #
##################################

def save_memories(file_path: str | os.PathLike, mem: associative_memory.AssociativeMemory) -> None:
  """Saves the memory to a pickle file.
  
  Args:
    file_path: The path of where to save the memory.
    mem: The memory to be saved."""
  with open(file_path, 'wb') as path:
    pickle.dump({
      'data': mem.get_data_frame()
    }, path)

def load_memories(file_path: str | os.PathLike, mem: associative_memory.AssociativeMemory | None = None, **kwargs) -> associative_memory.AssociativeMemory:
  """Load prebuilt memories from file.
  
  Args:
    file_path: The path of the file containing the memory data.
    mem: (Optional) An `AssociativeMemory` object into which memories can be loaded into.

  NOTE:
    `load_memories()` expects either keyword parameters necessary to construct a memory, or a prebuilt memory.
    If both are provided, the memory will be built using the prebuilt memory constructor. However, if no
    prebuilt memory is provided and no memory constructor parameters are provided, then this function will fail.
    """
  
  if mem is None:
    # If no memory was provided, kwargs are necessary to construct one
    mem_params = ['model', 'embedder', 'clock_now']
    DEFAULT_IMPORTANCE = 1.
    assert all(mem_param in kwargs.keys() for mem_param in mem_params), f"Missing memory parameters: {[mem_param for mem_param in mem_params if mem_param not in kwargs.keys()]}"
    if 'importance' not in kwargs:
      kwargs['importance'] = DEFAULT_IMPORTANCE
    mem = blank_memories.MemoryFactory(
      **kwargs
    ).make_blank_memory()
  else:
    # Otherwise, use the provided memory
    assert mem is not None, "If no memory parameters are provided, an `AssociativeMemory` object must be provided."
  
  with open(file_path, 'rb') as path:
    memory_bank = pickle.load(path)['data']

  for index, entry in memory_bank.iterrows():
    # Add each memory to the memory object.
    mem.add(
      text=entry.text,
      timestamp=entry.time,
      tags=entry.tags,
      importance=entry.importance
    )
  
  return mem

def save_component(file_path: str | os.PathLike, component: component.Component) -> None:
  """Save a component's state for reuse later."""
  with open(file_path, 'wb') as path:
    pickle.dump({
      'state': component.state(),
     }, path)
    
def load_component(file_path: str | os.PathLike, component: component.Component | None = None, **kwargs) -> component.Component:
  """Load a component's state from file.
  
  Args:
    file_path: The path of the file containing the component state.
    component: (Optional) An `AssociativeMemory` object into which memories can be loaded into.

  NOTE:
    `load_component()` expects either a component to be provided, or:
     1) The path to the component module to load (module), AND\n
     2) The name of the class to load (classname), AND\n
     3) A full list of parameters necessary to build the component, including optional ones (remaining kwargs).\n
     Building a new component from kwargs is discouraged, but it is possible.
  """
  
  if component is None:
    import warnings
    import importlib
    # Please don't do this
    warnings.formatwarning = lambda message, category, filename, lineno, line: '%s:%s: %s:%s\n' % (filename, lineno, category.__name__, message)
    warnings.warn(
      " Building a component from kwargs is discouraged and may be removed in the future.",
      category=FutureWarning
    )

    module = importlib.import_module(kwargs.pop("module"))
    component_class = getattr(module, kwargs.pop("classname"))

    # If no memory was provided, kwargs are necessary to construct one
    component_params = component_class.__init__.__code__.co_varnames
    assert all(param in kwargs.keys() for param in component_params if param != 'self'), f"Missing memory parameters: {[param for param in component_params if param not in kwargs.keys() and param != 'self']}"
    component = component_class(**kwargs)
  else:
    # Otherwise, use the provided memory
    assert component is not None, "If no component parameters are provided, a `component.Component` object must be provided."

  with open(file_path, 'rb') as path:
    state = pickle.load(path)['state']

  component._state = state

  return component

##################################
# endregion                      #
##################################

##################################
# region Decorators              #
##################################

def threading(func, enabled: bool = True):
  """
  A decorator to determine whether to use threading.
  """
  def wrapper(*args,  **kwargs):
    kwargs['threading'] = enabled
    return func(*args,  **kwargs)
  
class pipe(object):
    __name__ = "pipe"

    def __init__(self, function):
        self.function = function
        self.__doc__ = function.__doc__

        self.chained_pipes = []

    def __rshift__(self, other):
        assert isinstance(other, pipe)
        self.chained_pipes.append(other)
        return self

    def __rrshift__(self, other):

        result = self.function(other)

        for p in self.chained_pipes:
            result = p.__rrshift__(result)
        return result

    def __call__(self, *args, **kwargs):
        return pipe(lambda x: self.function(x, *args, **kwargs))
    
@pipe
def pprint(string: str) -> None:
  """Piped print function. Takes a string passed in with the right
  bit shift pipe and prints it."""
  print(string)

@pipe
def pipe_json(string: str) -> str:
  """Piped JSON format function. Takes a string passed in with the
  right bit shift pipe to be printed by the `pprint` function."""
  return json.dumps(string)

##################################
# endregion Decorators           #
##################################