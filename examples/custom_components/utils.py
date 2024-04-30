import os
import pickle

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
