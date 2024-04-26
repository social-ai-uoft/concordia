# Copyright 2023 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Agent component for self perception."""
import datetime
import re
import concurrent.futures
import json
import termcolor

from typing import Callable, Sequence
from collections import defaultdict
from retry import retry
from scipy.stats import zscore
from scipy.special import softmax

from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component


MAX_JSONIFY_ATTEMPTS = 5

class TPBComponent(component.Component):
  """Abstract class for a Theory of Planned Behaviour component."""

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
    self._agent_name = player_config.name
    self._goal = player_config.goal
    self._traits = player_config.traits
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = name
    self._json = []
    
  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state
  
  def json(self) -> list[dict]:
    return self._json
  
  def jsonify(self) -> list[dict]:
    """Take the output of the LLM and reformat it into a JSON array."""
    pass

class Behaviour(TPBComponent):
  """This component generates a list of candidate behaviours for an agent to take."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      num_behavs: int = 5,
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
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,
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
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,
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
    # Get all behaviours from the list
    behaviour_list = self._components[0].json()
    # Create output array
    output = []

    for behaviour, consequence_list in zip(behaviour_list, consequence_lists):

      # Split on positive/negative
      consequences = []
      for section in re.split(r'\*\*(?:Positive|Negative)(?: Consequences:|:)\*\*', consequence_list)[1:]:
        lines = [item.strip() for item in re.split(r'\d[\.:]\s', section) if item.strip()]
        for line in lines:
          # Check if two sets of parentheses are in the line
          if 0 not in [char in line for char in ["(", ")"]]:
            # Dictionary for each consequence including description, value, likelihood
            consequence = {}

            # Description precedes the parentheses
            consequence["description"] = re.search(
              r'^(.*?)(?=\()',
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
            consequences.append(consequence)
      # Add the behaviours and the consequences to the output array
      output.append({
        "behaviour": behaviour["behaviour"],
        "consequences": consequences,
        "attitude": self.eval_attitude(consequences)
      })
    return output
       
  @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def update(self) -> None:
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
          f"Double check that you did all of the behaviours and people correctly, for example that the numbers are all provided."
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
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,
                     num_memories_to_retrieve=num_memories_to_retrieve,verbose=verbose)
    
    self._num_people = num_people
    self._components = components

  def jsonify(self) -> list:
    
    # Split on behaviours
    people_lists = re.split(
      r"\n\nBEHAVIOUR\n\n",
      self._state
    )
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
        person["person"] = re.search(
            r'(.*?)(?=:|\(|\s-)',
            line
        ).group(1).replace("*", "").strip()
        person["approval"] = int(re.search(
            r'(?<=Approval:\s)(-?\d+)(?=\))',
            line
        ).group(1))
        behav_people.append(person)

      output.append({
        "behaviour": behaviour["behaviour"],
        "people": behav_people
      })

    return output

  @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def update(self) -> None:
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
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,
                     num_memories_to_retrieve=num_memories_to_retrieve,verbose=verbose)
    
    self._components = components

  def jsonify(self) -> list:

    motiv_list = self._state.split("\n\nPEOPLE\n\n")
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

  @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def update(self) -> None:

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
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,
                     num_memories_to_retrieve=num_memories_to_retrieve,verbose=verbose)
    
    self._components = components

  def eval_norm(self, people: list[dict]) -> int:
    """Generate a global subjective norm for a behaviour given a list of people.
    
    Args:
      people: A list of dictionaries listing the consequence description, value, and likelihood."""
    vs = [x["approval"] for x in people]
    ls = [x["motivation"] for x in people]
    return(sum([v * l for v, l in zip(vs, ls)]))
          

  @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
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
      clock_now: time callback to use for the state.
      num_memories_to_retrieve: Number of memories to retrieve.
      verbose: Whether to print the state.
    """

    # Initialize superclass
    super().__init__(name=name,model=model,memory=memory,player_config=player_config,clock_now=clock_now,
                     num_memories_to_retrieve=num_memories_to_retrieve,verbose=verbose)

    self._components = components

  def jsonify(self) -> list:
    """Take the output of the LLM and reformat it into a JSON array."""
    pass

  def update(self) -> None:

    self._jsons: list[list[dict]] = []
    self._json: list[dict] = []

    for component in self._components:
      if component.name == ["attitude", "norm"]:
        self._jsons.append(component.json())

    for _json in self._jsons:
      temp = defaultdict(set)
      for i in range(len(_json)):
        for k, v in _json[i].items():
          temp[k].add(v)
      self._json.append(temp)