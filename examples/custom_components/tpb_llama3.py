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

from typing import Callable, Sequence, Union
from collections import defaultdict
from retry import retry
from scipy.stats import zscore
from scipy.special import softmax

from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component

from examples.custom_components import utils
from examples.custom_components import plan as plan

MAX_JSONIFY_ATTEMPTS = 5

class TPBComponent(component.Component):
  """Abstract class for a Theory of Planned Behaviour component."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      memory_component: component.Component | None,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    
    """Initializes the abstract TPB component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      player_config: An AgentConfig object containing details about the agent.
      num_behavs: The number of behaviours to generate.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: Number of memories to retrieve.
      verbose: Whether to print the state.
    """

    self._verbose = verbose
    self._model = model
    self._memory = memory
    self._state = ''
    self._config = player_config
    self._agent_name = player_config.name
    self._goal = player_config.goal
    self._traits = player_config.traits
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = name
    self._json = []
    self._is_initialized = False
    self._has_memory_component = memory_component is not None
    self._memory_component = memory_component
    
  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state
  
  def json(self) -> list[dict]:
    return self._json
  
  def jsonify(self) -> list[dict]:
    """Take the output of the LLM and reformat it into a JSON array."""
    pass

  def _update(self) -> None:
    pass

  def update(self) -> None:
    if self._is_initialized:
      if self._verbose:
        print(termcolor.colored(f"{self._agent_name}'s {self.name()} component update:", color='light_green'))
      self._update()
      if self._state and self._has_memory_component:
        self._memory_component.observe(self._state)
    else:
      if self._verbose:
        print(termcolor.colored(f"{self._agent_name}'s {self.name()} component has been fully activated.", color='light_green'))
      self._is_initialized = True

class BasicEpisodicMemory(TPBComponent):
  """Basic component that synthesizes observations"""
  def __init__(
      self,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      clock_now: Callable[[], datetime.datetime] | None = None,
      timeframe: datetime.timedelta = datetime.timedelta(hours=1),
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
  
    super().__init__(name="memory",model=model,memory=memory,player_config=player_config,clock_now=clock_now,num_memories_to_retrieve=num_memories_to_retrieve,verbose=verbose)
    self._timeframe = timeframe

    # A dictionary inspired by SARSA containing the:
    # - Observation of preceding event by agent
    # - Deliberation/Cognition of agent
    # - Action of agent
    # - Observation of consequences/resulting event.
    self._working_memory = {
      "O": "",
      "D": "",
      "A": "",
      "O2": "",
      "R": ""
    }

  def observe(self, observation: str, wm_loc: str = "O2") -> None:
    """Take an observation and add it to the memory."""
    
    self._working_memory[wm_loc] = observation

    # If GM submits something, it is O2.
    # GM submits observations in the direct effect module.
    # We submit observations in the SequentialTPBModel: 
    # D after deliberation
    # A after action
    # O2 is moved to O after we get a new O2.
    
    if wm_loc == "O2":
      ltm_memory = (
        f"Initial state: {self._working_memory['O']}\n"
        f"Deliberation: {self._working_memory['D']}\n"
        f"Plan: {self._working_memory['A']}\n"
        f'Consequences: {self._working_memory["O2"]}\n'
        # f'Reflections: {self._working_memory["R"]}\n' # TODO: Reflection?
      )
      # Add the working memory to the LTM
      tags = ['observation']
      importance = 1.
      self._memory.add(f'[observation] {ltm_memory}', timestamp=self._clock_now(), tags=tags, importance=importance)
      # Move the resulting observation into the initial observation for the next state
      self._working_memory = {"O": f'{self._working_memory["O2"]}',"D": "","A": "","O2": ""}


  def summarize(self) -> str:
    """Summarize the agent's memory in the past timeframe."""

    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories_to_retrieve, add_time=True
        )
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f"Memories of {self._agent_name}:\n{mems}")

    question = (
      f"Given the memories provided, provide a summary of what has happened to "
      f"{self._agent_name} in the last {utils.format_timedelta(self._timeframe)}. "
    )

    summary = prompt.open_question(
      question,
      max_tokens=1000,
      max_characters=1200,
      terminators=()
    )

    return summary

class Behaviour(TPBComponent):
  """This component generates a list of candidate behaviours for an agent to take."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      num_behavs: int = 5,
      memory_component: component.Component | None = None,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    """Initializes the Behaviour component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      player_config: An AgentConfig object containing details about the agent.
      num_behavs: The number of behaviours to generate.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: Number of memories to retrieve.
      verbose: Whether to print the state.
    """

    # Initialize superclass
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,memory_component=memory_component,
                     num_memories_to_retrieve=num_memories_to_retrieve,verbose=verbose)
    
    self._num_behavs = num_behavs


  def jsonify(self) -> list:

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
    return behaviour_list
      
  @retry(AssertionError, tries = 5)
  def _update(self) -> None:
    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories_to_retrieve, add_time=True
        )
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f"Memories of {self._agent_name}:\n{mems}")

    if self._clock_now is not None:
      prompt.statement(f"Current time: {self._clock_now()}.\n")

    prompt.statement(f"Traits of {self._agent_name}: {self._traits}")
    prompt.statement(f"Goals of {self._agent_name}: {self._goal}")

    question = (
        f"Instructions: \n"
        f"Given the memories above, generate a list of {self._num_behavs} potential "
        f"behaviours above that {self._agent_name} can take in response to the situation. "
        f"Return the list of behaviours as a numbered list from 1 to {self._num_behavs}."
    )

    self._state = prompt.open_question(
      question,
      max_characters=5000,
      max_tokens=3000
    )
    
    self._json = self.jsonify()

    self._last_chain = prompt

    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

class Attitude(TPBComponent):
  """This component generates a personal attitude towards a given behaviour."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      components: Sequence[component.Component],
      memory_component: component.Component | None = None,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    """Initializes the Attitude component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      agent_name: Name of the agent.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: Number of memories to retrieve.
      verbose: Whether to print the state.
    """

    # Initialize superclass
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,memory_component=memory_component,
                     num_memories_to_retrieve=num_memories_to_retrieve,verbose=verbose)
    
    self._components = components

  def eval_attitude(self, consequences: list[dict]) -> int:
    """Generate a global attitude for a behaviour given a list of consequences.
    
    Args:
      consequences: A list of dictionaries listing the consequence description, value, and likelihood."""
    vs = [x["value"] for x in consequences]
    ls = [x["likelihood"] for x in consequences]
    return(sum([v * l for v, l in zip(vs, ls)]))

  def jsonify(self) -> list:

    # Split on behaviours
    consequence_lists = re.split(
      r"\n\nBEHAVIOUR\n\n",
      self._state
    )
    self._state = self._state.replace("\n\nBEHAVIOUR\n\n", "")

    # Get all behaviours from the list
    behaviour_list = self._components[0].json()
    # Create output array
    output = []

    for behaviour, consequence_list in zip(behaviour_list, consequence_lists):

      # Split on positive/negative
      consequences = []
      for section in re.split(r'\*\*(?:Positive|Negative)(?: Consequences:|:| Consequences)\*\*', consequence_list)[1:]:
        lines = [item.strip() for item in re.split(r'\n\d[\.:]\s', section) if item.strip()]
        for line in lines:
          # Check if two sets of parentheses are in the line
          if (0 not in [char in line for char in ["(", ")"]]) or (0 not in [char in line for char in [":", "*"]]):
            # Dictionary for each consequence including description, value, likelihood
            consequence = {}

            try:
              # Description precedes the parentheses
              consequence["description"] = re.search(
                r'^(.*?)(?=\(|\n)',
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
              print(termcolor.colored("Fake consequence detected!\n", color='red'))
              consequence["description"] = "Fake consequence!!!"
              consequence["value"] = 10
              consequence["likelihood"] = 100
            
            consequences.append(consequence)
      # Add the behaviours and the consequences to the output array
      output.append({
        "behaviour": behaviour["behaviour"],
        "consequences": consequences,
        "attitude": self.eval_attitude(consequences)
      })
    return output
       
  # @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def _update(self) -> None:
    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories_to_retrieve, add_time=True
        )
    )

    behavs = [item['behaviour'] for item in self._components[0].json()]

    def question(behav):
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(f"Memories of {self._agent_name}:\n{mems}")

      if self._clock_now is not None:
        prompt.statement(f"Current time: {self._clock_now()}.\n")

      prompt.statement(f"Traits of {self._agent_name}: {self._traits}")
      prompt.statement(f"Goals of {self._agent_name}: {self._goal}")

      q = (
          f"Instructions: \n"
          f"Given the memories above, and the candidate behaviour below: \n\n"
          f"{behav}\n\n"
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
          # f"This should be in the form of (Value: number, Likelihood: number) for each potential consequence. "
          # f"Here is an example: (Value: 8, Likelihood: 20)."
      )

      return prompt, prompt.open_question(
        q,
        max_characters=5000,
        max_tokens=3000
      )

    with concurrent.futures.ThreadPoolExecutor(max_workers = len(behavs)) as pool:
      prompts = []
      outputs = []
      for prompt, output in pool.map(
        question,
        behavs
      ):
        prompts.append(prompt)
        outputs.append(output)
      
    self._last_chain = prompts[-1]
    self._state = "\n\nBEHAVIOUR\n\n".join(outputs)

    self._json = self.jsonify()

    assert(len(self._json) > 0), "Did not successfully parse the array into a list of outputs."

    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

class People(TPBComponent):
  """This component generates a list of people who may have opinions about a behaviour."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      components: Sequence[component.Component],
      memory_component: component.Component | None = None,
      num_people: int = 5,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    """Initializes the People component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      player_config: An AgentConfig object containing details about the agent.
      components: A sequence including a Behaviour component.
      num_people: The number of people to generate.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: Number of memories to retrieve.
      verbose: Whether to print the state.
    """

    # Initialize superclass
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,memory_component=memory_component,
                     num_memories_to_retrieve=num_memories_to_retrieve,verbose=verbose)
    
    self._num_people = num_people
    self._components = components

  def jsonify(self) -> list:
    
    # Split on behaviours
    people_lists = re.split(
      r"\n\nBEHAVIOUR\n\n",
      self._state
    )
    self._state = self._state.replace("\n\nBEHAVIOUR\n\n", "")

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
          print(termcolor.colored("Fake person error!\n", color="red"))
          person["person"] = "Fake_Person"
          person["approval"] = 10
        behav_people.append(person)

      output.append({
        "behaviour": behaviour["behaviour"],
        "people": behav_people
      })

    return output

  # @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def _update(self) -> None:
    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories_to_retrieve, add_time=True
        )
    )

    behavs = [item['behaviour'] for item in self._components[0].json()]

    def question(behav):
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(f"Memories of {self._agent_name}:\n{mems}")

      if self._clock_now is not None:
        prompt.statement(f"Current time: {self._clock_now()}.\n")

      prompt.statement(f"Traits of {self._agent_name}: {self._traits}")
      prompt.statement(f"Goals of {self._agent_name}: {self._goal}")

      q = (
          f"Instructions: \n"
          f"Given the memories above, and the candidate behaviour below: \n\n"
          f"{behav}\n\n"
          f"Provide a numbered list of {self._num_people} people who might have opinions about whether "
          f"{self._agent_name} should do this behaviour or not. Do not include {self._agent_name} in "
          f"this list. For each person, include a rating from -10 to 10 indicating whether they approve "
          f"or disapprove of the behaviour. -10 is the most disapproval, and 10 is the most approval.\n"
          f"After listing each person, give their approval in the following format: (Approval: number).\n"
          f"Here is an example: Dave (Approval: 2)."
      )

      return prompt, prompt.open_question(
        q,
        max_characters=5000,
        max_tokens=3000
      )

    with concurrent.futures.ThreadPoolExecutor(max_workers = len(behavs)) as pool:
      prompts = []
      outputs = []
      for prompt, output in pool.map(
        question,
        behavs
      ):
        prompts.append(prompt)
        outputs.append(output)
      
    self._last_chain = prompts[-1]
    self._state = "\n\nBEHAVIOUR\n\n".join(outputs)

    self._json = self.jsonify()

    assert(len(self._json) > 0), "Did not successfully parse the array into a list of outputs."

    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

class Motivation(TPBComponent):
  """This gets the motivation of a character to take others' into account."""
  
  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      components: Sequence[component.Component],
      memory_component: component.Component | None = None,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    """Initializes the Motivation component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      player_config: An AgentConfig object containing details about the agent.
      components: A sequence including a Behaviour component.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: Number of memories to retrieve.
      verbose: Whether to print the state.
    """

    # Initialize superclass
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,memory_component=memory_component,
                     num_memories_to_retrieve=num_memories_to_retrieve,verbose=verbose)
    
    self._components = components

  def jsonify(self) -> list:

    motiv_list = self._state.split("\n\nPEOPLE\n\n")
    self._state = self._state.replace("\n\nPEOPLE\n\n", "")
    motivs = {}

    for motiv, person in zip(motiv_list, self._all_people):
      motivs[person.lower()] = int(re.search(
        r'(?<=Motivation: )(\d+)',
        motiv
      ).group(1)) / 100
    
    # Create a copy of the input json
    output = self._components[0].json()

    # For each behaviour...
    for i in range(len(output)):
      # For each person...
      for j in range(len(output[i]['people'])):
        # If the person has a motivation value...
        if output[i]['people'][j]['person'].lower() in motivs.keys():
          # Add it to the json
          output[i]['people'][j]['motivation'] = motivs[output[i]['people'][j]['person'].lower()]
    
    return output

  # @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def _update(self) -> None:

    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories_to_retrieve, add_time=True
        )
    )

    # This horrible expression generates a list of uniquely identified people across all behaviours.
    self._all_people = list(set.union(*map(set, ([j['person'] for j in i] for i in [item['people'] for item in self._components[0].json()]))))

    def question(person):
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(f"Memories of {self._agent_name}:\n{mems}")

      if self._clock_now is not None:
        prompt.statement(f"Current time: {self._clock_now()}.\n")

      prompt.statement(f"Traits of {self._agent_name}: {self._traits}")
      prompt.statement(f"Goals of {self._agent_name}: {self._goal}")

      q = (
          f"Instructions: \n"
          f"Given the memories above, and the person below: \n\n"
          f"{person}\n\n"
          f"Indicate how motivated {self._agent_name} is to take into consideration {person}'s approval or disapproval. "
          f"In other words, how much will {self._agent_name} consider whether {person} approves or disapproves "
          f"of a behaviour when {self._agent_name} is considering how to act in a given situation."
          f"Provide a value from 0 to 100, with 0 being not at all, and 100 being the most infuential, "
          f"in the format (Motivation: number). Remember to consider the full scale from 0 to 100 in the ratings. "
          f"Do not provide any explanation or description."
      ) 

      return prompt, prompt.open_question(
        q,
        max_characters=5000,
        max_tokens=3000
      )

    with concurrent.futures.ThreadPoolExecutor(max_workers = len(self._all_people)) as pool:
      prompts = []
      outputs = []
      for prompt, output in pool.map(
        question,
        self._all_people
      ):
        prompts.append(prompt)
        outputs.append(output)
      
    self._last_chain = prompts[-1]
    self._state = "\n\nPEOPLE\n\n".join(outputs)

    self._json = self.jsonify()

    assert(len(self._json) > 0), "Did not successfully parse the array into a list of outputs."

    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

class SubjectiveNorm(TPBComponent):
  """This computes the subjective norms applied to a behaviour."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      components: Sequence[component.Component],
      memory_component: component.Component | None = None,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    """Initializes the Motivation component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      player_config: An AgentConfig object containing details about the agent.
      components: A sequence including a Behaviour component.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: Number of memories to retrieve.
      verbose: Whether to print the state.
    """

    # Initialize superclass
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,memory_component=memory_component,
                     num_memories_to_retrieve=num_memories_to_retrieve,verbose=verbose)
    
    self._components = components

  def eval_norm(self, people: list[dict]) -> int:
    """Generate a global subjective norm for a behaviour given a list of people.
    
    Args:
      people: A list of dictionaries listing the consequence description, value, and likelihood."""
    vs = [x["approval"] for x in people]
    ls = [x["motivation"] for x in people]
    return(sum([v * l for v, l in zip(vs, ls)]))
          

  # @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def update(self) -> None:
    
    # Take the motivation json
    output = self._components[0].json()
    for i in range(len(output)):
      output[i]['norm'] = self.eval_norm(output[i]['people'])

    self._json = output

class TPB(TPBComponent):
  """Full TPB model."""
  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      components: Sequence[TPBComponent],
      memory_component: component.Component | None = None,
      w: float = 0.5,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    """Initializes the TPB component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      player_config: An AgentConfig object containing details about the agent.
      components: A sequence including a Behaviour component.
      w: The weight parameter. Default is 0.5 (equal weight given to attitudes and subjective norms). 
      A higher value indicates more weight towards attitudes, lower indicates more weight towards subjective norms.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: Number of memories to retrieve.
      verbose: Whether to print the state.
    """

    # Initialize superclass
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,memory_component=memory_component,
                     num_memories_to_retrieve=num_memories_to_retrieve,verbose=verbose)

    self._w = w
    self._components = {}
    for comp in components:
      self.add_component(comp)

  def add_component(self, comp: component.Component) -> None:
    """Add a component."""
    if comp.name() in self._components:
      raise ValueError(f'Duplicate component name: {comp.name()}')
    else:
      self._components[comp.name()] = comp

  def collate(self, measure: str) -> list:

    if measure == "behaviour":
      return [re.search(r'(.*?)(?=:)', behaviour["behaviour"]).group(1).replace('*', '').strip() if re.search(r'(.*?)(?=:)', behaviour["behaviour"]) is not None else behaviour["behaviour"].replace("*", "").strip() for behaviour in self._components["attitude"].json()]
    else:
      return [behaviour[measure] for behaviour in self._components[measure].json()]
    
  def evaluate_intentions(self) -> np.ndarray:
    """
    Compute the behavioural intentions.
    """
    attitudes = zscore(self.collate("attitude"))
    norms = zscore(self.collate("norm"))
    
    w = self._w
    # Weigh the two values
    behavioural_intentions = (w * attitudes) + ((1 - w) * norms)
    # Compute softmax
    behav_probs = softmax(behavioural_intentions)
    return behav_probs
  
  def evaluate_probability_of_behaviour(self, behaviour: int | str) -> float:
    """Compute the probability of a behaviour.
    
    Args:
      behaviour: An integer indicating the index of the behaviour or a string matching the description of the behaviour."""
    
    if isinstance(behaviour, str):
      behaviours = self.collate("behaviour")
      index = behaviours.index(behaviours)
    else:
      index = behaviour
    
    return self.evaluate_intentions()[index]
  
  def plot(self, file_path: Union[str, os.PathLike] | None = None) -> None:
    """Plot the outputs.
    
    Args:
      file_path: An optional string or PathLike indicating the location to save the file."""
    import matplotlib.pyplot as plt

    behaviours = self.collate("behaviour")
    attitudes = self.collate("attitude")
    norms = self.collate("norm")
    behav_probs = self.evaluate_intentions()

    bw = 0.25

    b1 = np.arange(len(attitudes))
    b2 = [x + bw for x in b1]
    b3 = [x + bw for x in b2]

    plt.figure(figsize=(8,8))

    plt.barh(b3, softmax(attitudes), height = bw, color = 'darkgreen', label = 'Attitudes')
    plt.barh(b2, softmax(norms), height = bw, color = 'limegreen', label = 'Subjective Norms')
    plt.barh(b1, behav_probs, height = bw, color = 'forestgreen', label = 'Behavioural Intentions')

    plt.ylabel('Behaviours')
    plt.xlabel('Action Probability')
    plt.yticks([x + bw for x in b1], behaviours)
    plt.xlim((0, 1))
    # plt.yticks(rotation=90)
    # plt.subplots_adjust(bottom=0.50)
    plt.tight_layout()
    plt.legend(loc="upper right")
    if file_path is not None:
      plt.savefig(file_path, dpi = 150)

    plt.show()

  def update(self) -> None:
    if self._is_initialized:
      if self._verbose:
        print(termcolor.colored(f"{self._agent_name}'s {self.name()} component update:", color='light_green'))
      self._update()
    else:
      if self._verbose:
        print(termcolor.colored(f"{self._agent_name}'s {self.name()} component has been fully activated.", color='light_green'))
      self._is_initialized = True
      print(termcolor.colored(
      'Using the thin goal...', color="light_magenta"
      ))
      self._components["thin_goal"].update()
      self._state = self._components["thin_goal"].state()

  def _update(self) -> None:

    print(termcolor.colored(
      'Initializing _update()...', color="light_magenta"
    ))

    attitudes = self.collate("attitude")
    norms = self.collate("norm")

    # Weighting factor
    behav_probs = self.evaluate_intentions()

    # Choose behaviour
    behaviours = self.collate("behaviour")
    chosen_behav = np.random.choice(behaviours, p=behav_probs)

    self._state = (
      f'After considering {utils.pronoun(self._config, case = "genitive")} options, '
      f"{self._agent_name}'s current goal is to successfully accomplish or complete the following behaviour: {chosen_behav}."
    )

    if self._verbose:
      for i in range(len(behaviours)):
        print(termcolor.colored(f"Behaviour: {behaviours[i]}.", color="light_magenta"))
        print(termcolor.colored(f"Attitude: {round(attitudes[i], 2)}. Norm: {round(norms[i], 2)}. Action probability: {round(behav_probs[i], 2)}", color="light_magenta"))
      print(termcolor.colored(self._state, 'light_magenta'), end='')

class ThinGoal(TPBComponent):
  """ThinGoal outputs a goal based on the player configuration goal."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    
    """Initializes the ThinGoal component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      player_config: An AgentConfig object containing details about the agent.
      num_behavs: The number of behaviours to generate.
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: Number of memories to retrieve.
      verbose: Whether to print the state.
    """
    
    super().__init__(name, model, memory, player_config, clock_now, num_memories_to_retrieve, verbose)
    self._is_initialized = True

  def update(self) -> None:

    mems = '\n'.join(
          self._memory.retrieve_recent(
              self._num_memories_to_retrieve, add_time=True
          )
      )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f"Memories of {self._agent_name}:\n{mems}")

    if self._clock_now is not None:
      prompt.statement(f"Current time: {self._clock_now()}.\n")

    prompt.statement(f"Traits of {self._agent_name}: {self._traits}")
    prompt.statement(f"Goals of {self._agent_name}: {self._goal}")

    question = (
        f"Instructions: \n"
        f"Given the memories above, restate {self._agent_name}'s goal in one sentence."
    )

    self._state = prompt.open_question(
      question,
      max_characters=5000,
      max_tokens=3000
    )
    
    self._json = self.jsonify()

    self._last_chain = prompt

    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

class SequentialTPBModel(component.Component):
  """
  Explicit sequential model for the Theory of Planned Behaviour.
  """
  def __init__(
      self,
      name: str,
      components: Sequence[component.Component],
  ):
    
    self._name = name
    self._components = {}
    for component in components:
      self._components[component.name()] = component

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  def observe(self, observation: str) -> None:
    self._components["memory"].observe(observation)

  def update(self) -> None:

    # Update the components one by one, in order.

    # First, the TPB components...
    self._components["behaviour"].update()
    self._components["attitude"].update()
    self._components["people"].update()
    self._components["motivation"].update()
    self._components["norm"].update()

    # Store the deliberation summaries from all of the TPB components...
    deliberation = self._components["observation"].summarize({
      "behaviour", "attitude", "norm"
    })
    # and then put them into the working memory as the D component
    self._components["memory"].observe(deliberation, wm_loc = "D")

    # After deliberations are complete, synthesize the TPB model into the plan
    self._components["tpb"].update()
    self._components["situation"].update()
    self._components["Plan"]._last_update = None # Set last update to none so it always replans.
    self._components["Plan"].update()
    self._state = self._components["Plan"].state()
    
    # Add the plan as the A component of the working memory
    self._components["memory"].observe(self._state, wm_loc = "A")
