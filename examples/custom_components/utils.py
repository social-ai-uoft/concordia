import os
import pickle
import datetime

from typing import Union

from concordia.agents import basic_agent
from concordia.associative_memory import formative_memories, associative_memory, blank_memories

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

  divs = {
    "seconds": _second_div,
    "minutes": _minute_div,
    "hours": _hour_div,
    "days": _day_div
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

def save_memories(mem: associative_memory.AssociativeMemory, file_path: Union[str, os.PathLike]) -> None:

  with open(file_path, 'wb') as path:
    pickle.dump({
      'data': mem.get_data_frame()
    }, path)

def load_memories(mem: associative_memory.AssociativeMemory, file_path: Union[str, os.PathLike]) -> associative_memory.AssociativeMemory:

  with open(file_path, 'rb') as path:
    memory_bank = pickle.load(path)['data']
  
  for index, entry in memory_bank.iterrows():

    mem.add(
      text=entry.text,
      timestamp=entry.time,
      tags=entry.tags,
      importance=entry.importance
    )
  
  return mem
