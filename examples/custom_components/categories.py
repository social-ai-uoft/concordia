"""Component for parsing generic language."""
import datetime
from typing import Callable
import json

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor

class CategoryConcept(component.Component):
  """Describes category concept for the generic agent"""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      
      mem_num: int = 100,
      verbose: bool = False
  ):
    
    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._state = ''
    self._agent_name = agent_name
    self._mem_num = mem_num
    self._name = name

  def name(self) -> str:
    return self._name
  
  def state(self) -> str:
    return self._state
  
  def update(self) -> None:
    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._mem_num, add_time=True
        )
    )
    
    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

    question = (
      f'Given the memories above, what does {self._agent_name} think is true about Zarpies?'
    )

    self._state = prompt.open_question(
        question,
        answer_prefix=f'{self._agent_name} thinks that ',
        max_characters=3000,
        max_tokens=1000,
    )

    self._state = f'{self._agent_name} thinks that {self._state}'

    self._last_chain = prompt
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')