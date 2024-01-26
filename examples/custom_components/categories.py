"""Component for parsing generic language."""
import datetime
from typing import Callable, Dict, List, Any
import json
import random

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
      question_bank: dict,
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
    self._question_bank = question_bank

  def name(self) -> str:
    return self._name
  
  def state(self) -> str:
    return self._state
  
  def observe(self, observation: str) -> None:
    """See base class."""
    
    for question in self._question_bank["questions"]:

      # Multiple choice question
      prompt = interactive_document.InteractiveDocument(self._model)

      letters = ['a', 'b', 'c', 'd']
      options = '\n'.join([f'({letters[i]}) {question["options"][i]}' for i in range(len(question["options"]))])
      answer = prompt.open_question(
        question=f'\n{observation}\n{question["question"]}\n\nOptions: {options}\nAfter selecting your choice, please explain why you chose it.',
        terminators=[]
      )

      if self._verbose:
        print(
          termcolor.colored(f'Question: {question["question"]}\nAnswer: {answer}\n', color='blue'), end = '\n'
        )

      # answer = prompt.multiple_choice_question(
      #   question=f'\n{observation}\n{question["question"]}',
      #   answers=question["options"]
      # )
      
      # if self._verbose:
      #   print(
      #     termcolor.colored(f'Question: {question["question"]}\nAnswer: {question["options"][answer]}', color='blue'), end = '\n'
      #   )

    # self._last_chain = prompt
    # if self._verbose:
    #   print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

def sample_statements(
    condition: str, 
    experiment: Dict[str, Dict[str, List[Any]]], 
    prop_physical: float = 0.5,
    num_statements = 5
) -> str:
  """Take a dictionary containing three different entries each containing an
    array of different statements that vary by condition, and sample a set of
    statements from the given condition without replacement.

  Parameters:
    condition: The condition to draw the samples from.
    experiment: The dictionary in which the samples are located.
    prop_physical: Proportion of statements to be drawn from "physical" list.
    num_statements_phys: Number of statements to draw from the "physical" list.
    num_statements_behav: Number of statements to draw from the "behavioral" list.
    
  Returns:
    A multi-line string containing a bulleted list of the statements to be 
    presented."""
    
  # Check if condition is in the keys of experiment dictionary
  assert condition in experiment.keys(), f"Condition '{condition}' not found in experiment."

  # Ensure prop_physical is between 0 and 1
  assert 0 <= prop_physical <= 1, "Proportion of physical statements must be between 0 and 1."
  
  # Calculate the number of statements to draw from each list based on proportion
  num_statements_phys = round(prop_physical * (num_statements))
  num_statements_behav = num_statements - num_statements_phys
  
  # Randomly select statements from the "physical" and "behavioral" lists without replacement
  statements_phys = random.sample(experiment[condition]["physical"], num_statements_phys)
  statements_behav = random.sample(experiment[condition]["behavioral"], num_statements_behav)
  
  # Combine the two lists of statements and create a bulleted list string to be returned
  statements = statements_phys + statements_behav
  random.shuffle(statements)
  bulleted_list = "\n".join([f"* {statement}" for statement in statements])

  return bulleted_list