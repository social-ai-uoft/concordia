# Author: Rebekah Gelpi
# Concordia Cognitive Model: Theory of Planned Behaviour

"""Agent component for self perception."""
import datetime
import re
import concurrent.futures
import json
import os
import termcolor
import numpy as np

from itertools import repeat
from retry import retry
from scipy.stats import zscore
from typing import Any, Callable, Sequence

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component

from examples.tpb import utils
from examples.tpb import memory as tpb_memory

MAX_JSONIFY_ATTEMPTS = 5

class TPBComponent(component.Component):
  """Theory of Planned Behaviour component shared structure."""
  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      config: tpb_memory.TPBAgentConfig,
      components: Sequence[component.Component] | None = None,
      num_memories: int = None,
      clock_now: Callable[[], datetime.datetime] | None = None,
      verbose: bool = False,
      **kwargs
  ):
    """Initialize a base TPB component.
    
    Args:
      name: The name of the component.
      model: The language model.
      memory: The agent memory.
      clock_now: Time callback function.
      verbose: Whether to print the state.
    """
    self._name = name
    self._model = model
    self._config = config
    self._memory = config.memory
    self._agent_name = config.name
    self._num_memories = num_memories
    self._components = components
    self._clock_now = clock_now
    self._verbose = verbose
    self._state = ''
    self._last_chain = interactive_document.InteractiveDocument(model=model)
    self._json = []
    self._is_initialized = False
    self._kwargs = kwargs
    self._print = lambda *args : print(termcolor.colored('[LOG] [Component] ' + " ".join(args).replace("\n", "\n[LOG] [Component] "), color='green'))
    self._warn = lambda *args : print(termcolor.colored('[WARN] [Component] ' + " ".join(args), color='yellow'))
    self._err = lambda *args : print(termcolor.colored('[ERR] [Component] ' + " ".join(args), color='red'))

  def name(self) -> str:
    """Return the component name."""
    return self._name

  def state(self) -> str:
    """Return the component's current state."""
    return self._state

  def initialize(self) -> None:
    """Initialize, so that the component is not run the first time it is called."""
    if self._verbose:
      self._print(f"{self.name()} initialized.")
    self._is_initialized = True
  
  def json(self) -> list[dict]:
    """Return the state as a JSON array."""
    return self._json
  
  def jsonify(self) -> None:
    """Convert the state into a JSON array."""
    pass

  def prompt_builder(
      self,
      prompt: interactive_document.InteractiveDocument,
      statements: dict[str, Any]
  ) -> None:
    """Given a document prompt and a sequence of statements,
    make all prompt statements."""

    for k, v in statements.items():
      if isinstance(v, datetime.datetime): # Special case: clock now
        prompt.statement(f"{k}: {v}") 
      elif v is not None: # Base case
        prompt.statement(f"{k} of {self._config.name}: {v}")

  def prompt_batch(
      self, 
      inputs: list[str], 
      func: Callable[[str], tuple], 
    ):
    """Batch prompts into a single-threaded for loop
    or a multi-threaded executor."""
    if not self._kwargs['threading']:
      prompts = []
      outputs = []
      for i in inputs:
        prompt, output = func(i)
        prompts.append(prompt)
        outputs.append(output)
    else:
      with concurrent.futures.ThreadPoolExecutor(max_workers = len(inputs)) as pool:
        prompts = []
        outputs = []
        for prompt, output in pool.map(
          func,
          inputs
        ):
          prompts.append(prompt)
          outputs.append(output)      
    return prompts, outputs

  def question(
      self,
      question: str,
  ) -> tuple[interactive_document.InteractiveDocument, str]:
    """
    Open an InteractiveDocument and pose a question to it.

    Args:
      question (str): The question to ask the LLM.
    Returns:
      tuple[InteractiveDocument, str]: The prompt and the response.
    """
    
    mems = '\n'.join(
      self._memory.retrieve_recent(
        self._num_memories, add_time=True
      )
    )
        
    prompt = interactive_document.InteractiveDocument(self._model)
    self.prompt_builder(prompt, {
      "Memories": "\n" + mems,
      "Traits": self._config.traits,
      "Goals": self._config.goal,
      "Current time": self._clock_now()
    })

    return prompt, prompt.open_question(
      question,
      max_characters=5000,
      max_tokens=3000
    )
    
  def _update(self) -> None:
    pass

  def update(self) -> None:
    """Update the component."""
    if self._is_initialized:
      self._update()
      if self._verbose:
        self._print(self._last_chain.view().text())
    else:
      self.initialize()

class Behaviour(TPBComponent):
  """Behaviour pipeline for the TPB."""
  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      config: tpb_memory.TPBAgentConfig,
      num_memories: int = 100,
      num_behavs: int = 5,
      clock_now: Callable[[], datetime.datetime] | None = None,
      verbose: bool = False
   ):
    """
    Initializes the Behaviour component.
    
    Args:
      name: The name of the component.
      model: Language model.
      config: Agent configuration class.
      num_memories: The number of memories to retrieve.
      num_behavs: The number of behaviours to generate.
      clock_now: Callback for the game clock.
      verbose: Whether to print the state."""
    super().__init__(name=name, model=model, config=config, num_memories=num_memories, 
                     clock_now=clock_now, verbose=verbose)
    self._num_behavs = num_behavs
  
  def jsonify(self) -> None:
    
    behaviour_list = []
    # Split on each digit
    lines = re.split(r'\d[\.:]\s?', self._state)
    # Make sure this has returned a numbered list, or throw an assertion error
    assert len(lines) > 1, "LLM did not generate a numbered list of behaviours."
    # Add the behaviour to the line.
    for line in lines[1:]:
      item = {}
      item['behaviour'] = line.strip()
      behaviour_list.append(item)

    self._json = behaviour_list

  @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def _update(self):

    question = (
      "Instructions: \n"
      f"Given the memories above, generate a list of {self._num_behavs} potential "
      f"behaviours above that {self._agent_name} can take in response to the situation. "
      f"Return the list of behaviours as a numbered list from 1 to {self._num_behavs}."
    )

    prompt, self._state = self.question(
      question,
    )

    self._last_chain = prompt

    # Convert the state to a JSON.
    self.jsonify()
  
class Attitude(TPBComponent):
  """This component generates a personal attitude towards a given behaviour."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      config: tpb_memory.TPBAgentConfig,
      components: Sequence[component.Component],
      num_memories: int = 100,
      clock_now: Callable[[], datetime.datetime] | None = None,
      verbose: bool = False,
      **kwargs
   ):
    """
    Initializes the Behaviour component.
    
    Args:
      name: The name of the component.
      model: Language model.
      config: Agent configuration class.
      num_memories: The number of memories to retrieve.
      num_behavs: The number of behaviours to generate.
      clock_now: Callback for the game clock.
      verbose: Whether to print the state."""

    # Initialize superclass
    super().__init__(name=name, model=model, config=config, num_memories=num_memories, 
                     clock_now=clock_now, verbose=verbose, components=components, **kwargs)

  def eval_attitude(self, consequences: list[dict]) -> int:
    """Generate a global attitude for a behaviour given a list of consequences.
    
    Args:
      consequences: A list of dictionaries listing the consequence description, value, and likelihood."""
    vs = [x["value"] for x in consequences]
    ls = [x["likelihood"] for x in consequences]
    return(sum([v * l for v, l in zip(vs, ls)]))

  def jsonify(self) -> None:

    # Split on behaviours
    consequence_lists = re.split(
      r"\n\n####\n\n",
      self._state
    )
    self._state = self._state.replace("\n\n####\n\n", "")

    # Get all behaviours from the list
    behaviour_list = self._components[0].json()
    # Create output array
    output = []

    for behaviour, consequence_list in zip(behaviour_list, consequence_lists):

      # Split on positive/negative
      consequences = []
      for section in re.split(r'(\**?)(?:[Pp]ositive|[Nn]egative)(?: [Cc]onsequences:|:| [Cc]onsequences)(\**?)', consequence_list)[1:]:
        lines = [item.strip() for item in re.split(r'\n\d[\.:]\s', section) if item.strip()]
        for line in lines:
          # Check if two sets of parentheses are in the line
          if (0 not in [char in line for char in ["(", ")"]]) or (0 not in [char in line for char in [":", "*"]]) or (0 not in [char in line for char in ["Value", "Likelihood"]]):
            # Dictionary for each consequence including description, value, likelihood
            consequence = {}

            try:
              # Description precedes the parentheses
              consequence["description"] = re.search(
                r'^(.*?)(?=\(|\n|Value:)',
                line
              ).group(1).strip()

              # Value follows "Value: "
              consequence["value"] = int(re.search(
                r'(?<=Value:\s)(-?\d+)',
                line
              ).group(1))

              # Likelihood follows "Likelihood: "
              consequence["likelihood"] = int(re.search(
                r'(?<=Likelihood:\s)(\d+)(%?)',
                line
              ).group(1)) / 100
            except AttributeError:
              self._warn(f"The line: {line} could not be parsed using the regex syntax. Retrying.")
              raise AssertionError("Unable to parse the line. Retrying.")
            
            consequences.append(consequence)
      # Add the behaviours and the consequences to the output array
      output.append({
        "behaviour": behaviour["behaviour"],
        "consequences": consequences,
        "attitude": self.eval_attitude(consequences)
      })
    
    self._json = output
       
  @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def _update(self) -> None:
    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories, add_time=True
        )
    )

    behavs = [item['behaviour'] for item in self._components[0].json()]

    q = (
          f"Instructions: \n"
          f"Given the memories above, and the candidate behaviour below: \n\n"
          "{0}\n\n"
          f"List three potential positive and three potential negative consequence of the behaviour for {self._agent_name}. "
          f"Only list consequences that directly affect whether the behaviour would be good or bad for {self._agent_name}. "
          f"For each potential consequence, indicate with a number from -10 to 10 how bad to good that consequence is for "
          f"{self._agent_name}. -10 is the worst possible outcome, 0 is neutral, and 10 is the best possible outcome. "
          f"Not all consequences are likely to occur if a behaviour is done. For each potential consequence, "
          f"indicate with a number from 0 to 100 how likely that consequence is to occur. 0 is impossible, 100 is certain. "
          f"List the likelihood of each consequences if the behaviour is done after each of the separate consequences to each behaviour. "
          f"This is not the likelihood that {self._agent_name} will do the task, but rather the likelihood that the consequence will occur if the task is done. "
          f"So, each of the 6 potential separate consequences should have a likelihood value. "
          f"This should be in the form of (Value: number, Likelihood: number) for each potential consequence, "
          f"Remember, there should be three separate positive and three negative consequences for the potential behaviour "
          f"each with its own value and likelihood.\n"
          f"Double check that you did all of the behaviours and people correctly, for example that the numbers are all provided. "
          f"This should be in the form of (Value: number, Likelihood: number) for each potential consequence. "
          f"Here is an example: (Value: 8, Likelihood: 20)."
      )

    prompts, outputs = self.prompt_batch([q.format(behav) for behav in behavs], self.question)

    self._last_chain = prompts[-1]
    self._state = "\n\n####\n\n".join(outputs)

    self.jsonify()

    assert(len(self._json) > 0), "Did not successfully parse the array into a list of outputs."


class People(TPBComponent):
  """This component generates a list of people who may have opinions about a behaviour."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      config: tpb_memory.TPBAgentConfig,
      components: Sequence[component.Component],
      num_people: int = 5,
      num_memories: int = 100,
      clock_now: Callable[[], datetime.datetime] | None = None,
      verbose: bool = False,
      **kwargs
   ):
    """
    Initializes the People component.
    
    Args:
      name: The name of the component.
      model: Language model.
      config: Agent configuration class.
      num_memories: The number of memories to retrieve.
      num_behavs: The number of behaviours to generate.
      clock_now: Callback for the game clock.
      verbose: Whether to print the state."""

    # Initialize superclass
    super().__init__(name=name, model=model, config=config, num_memories=num_memories, 
                     clock_now=clock_now, verbose=verbose, components=components)
    self._kwargs = kwargs
    self._num_people = num_people

  def jsonify(self) -> list:
    
    # Split on behaviours
    people_lists = re.split(
      r"\n\n####\n\n",
      self._state
    )
    self._state = self._state.replace("\n\n####\n\n", "")

    # Get all behaviours from the list
    behaviour_list = self._components[0].json()
    # Create output array
    output = []

    for behaviour, people_list in zip(behaviour_list, people_lists):

      behav_people = []

      lines = (
        [item.strip() for item in re.split(r'\d\.\s', people_list) if 0 not in [char in item for char in ["(", ")"]]]
      )
      for line in lines:
        person = {}
        try:
          person["person"] = re.search(
              r'(.*?)(?=:|\(|\s-)',
              line
          ).group(1).replace("*", "").strip()
          person["approval"] = int(re.search(
              r'(?<=Approval:\s)(-?\d+)(?=\))',
              line
          ).group(1))
        except AttributeError:
          self._warn("No person found in line: {}".format(line))
          raise AssertionError("No person found in line")
        behav_people.append(person)

      output.append({
        "behaviour": behaviour["behaviour"],
        "people": behav_people
      })

    return output

  @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def _update(self) -> None:

    q = (
        f"Instructions: \n"
        f"Given the memories above, and the candidate behaviour below: \n\n"
        "{0}\n\n"
        f"Provide a numbered list of {self._num_people} people who might have opinions about whether "
        f"{self._agent_name} should do this behaviour or not. Do not include {self._agent_name} in "
        f"this list. For each person, include a rating from -10 to 10 indicating whether they approve "
        f"or disapprove of the behaviour. -10 is the most disapproval, and 10 is the most approval.\n"
        f"After listing each person, give their approval in the following format: (Approval: number).\n"
        f"Here is an example: Dave (Approval: 2)."
    )

    behavs = [item['behaviour'] for item in self._components[0].json()]

    #####################################################
    # region: Optional single or multi-thread execution #
    #####################################################
    if 'threading' in self._kwargs and self._kwargs['threading'] == "DISABLED":
      prompts = []
      outputs = []
      for behav in behavs:
        prompt, output = self.question(q.format(behav))
        prompts.append(prompt)
        outputs.append(output)
    else:
      with concurrent.futures.ThreadPoolExecutor(max_workers = len(behavs)) as pool:
        prompts = []
        outputs = []
        for prompt, output in pool.map(
          self.question,
          [(q.format(behav)) for behav in behavs]
        ):
          prompts.append(prompt)
          outputs.append(output)
    #####################################################
    # endregion                                          #
          behavs,
        ):
          prompts.append(prompt)
          outputs.append(output)
    #####################################################
    # endregion                                         #
    #####################################################
      
    self._last_chain = prompts[-1]
    self._state = "\n\n####\n\n".join(outputs)

    self._json = self.jsonify()

    assert(len(self._json) > 0), "Did not successfully parse the array into a list of outputs."