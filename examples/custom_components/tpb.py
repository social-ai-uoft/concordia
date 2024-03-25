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
import concurrent
import json
from typing import Callable, Sequence

from concordia.associative_memory import associative_memory
from concordia.associative_memory import formative_memories
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor

MAX_JSONIFY_ATTEMPTS = 5

class BehavioralChoices(component.Component):
  """This component generates a list of candidate behaviours for an agent to take."""

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
    """Initializes the BehavioralChoices component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      agent_name: Name of the agent.
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
    self._list = []

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state
  
  def jsonify(self) -> list[dict]:

    list_1 = re.split(
      r'\*\*Potential Behavi(?:ou|o)r',
      self._state
    )

    all_behavs = []

    for behavs in list_1[1:]:
      each_behav = {}
      list_2 = re.split(
        r'\*\*(?:Positive|Negative)(?: Consequences:|:)\*\*',
        behavs
      )

      list_2 = [item.replace(r'*', '').replace('    ', '').strip() for item in list_2]


      try:
        each_behav["behaviour"] = re.search(
          r'(?<=\d:\s)(.*)',
          list_2[0]
        ).group(1)
      except AttributeError:
        each_behav["behaviour"] = re.search(
          r'(?<=: )(.*)',
          list_2[0]
        ).group(1)

      each_behav["consequences"] = []

      for consequences in list_2[1:4]:
        consequence_list = re.split(
          r'\n',
          consequences
        )
        
        for consequence in consequence_list:

          each_consequence = {}


          if consequence.__contains__("("):
            print(f"- {consequence} -")

            each_consequence["description"] = re.sub(
              r'\d\. ',
              '',
              re.search(
                r'(.*)(?=\()',
                consequence
              ).group(1).strip()
            )
            
            each_consequence["value"] = int(re.search(
              r'(?<=Value: )(-?\d+)(?=,)',
              consequence
            ).group(1))
            each_consequence["likelihood"] = float(re.search(
              r'(?<=Likelihood: )(\d+)(%?)(?=\))',
              consequence
            ).group(1).strip("%"))/100
            # print(each_consequence)

            each_behav["consequences"].append(each_consequence)
      
      all_behavs.append(each_behav)

    return all_behavs

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
        f"Consider each of the potential behaviours above that {self._agent_name} can take in response to the situation. "
        # f"Generate a list of five potential behaviours that {self._agent_name} can take in response to the situation. "
        f"For each potential behaviour, provide three possible positive and three possible negative consequences of that behaviour. "
        f"For each potential consequence, indicate with a number from -10 to 10 how bad to good "
        f"that consequence is for {self._agent_name}. -10 is the worst possible outcome, 0 is neutral, and 10 is the best possible outcome. "
        f"List the positive or negative value after each of the separate consequences to each behaviour. "
        f"For each potential behaviour, provide just the consequence numbers separated by a comma. "
        f"For each potential consequence, indicate with a number from 0 to 100 how likely that consequence is to occur. 0 is impossible, 100 is certain. "
        f"For each behaviour, a set of possible consequences are now given. yet, not all consequences are likely to occur if a behaviour is done. "
        f"List the likelihood of each consequences if the behaviour is done after each of the separate consequences to each behaviour. "
        f"This is not the likelihood that {self._agent_name} will do the task, but rather the likelihood that the consequence will occur if the task is done. "
        f"So, each of the 6 potential separate consequences should have a likelihood value. "
        f"This should be in the form of (Value: number, Likelihood: number) for each potential consequence, "
        f"Remember, there should be three separate positive and three negative consequences for each potential behaviour "
        f"each with its own value and likelihood.\n"
        f"double check that you did all of the behaviours and people correctly, for example that the numbers are all provided."
    )


    self._state = prompt.open_question(
      question,
      max_characters=5000,
      max_tokens=3000,
      terminators=None
    )

    self._last_chain = prompt

    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

class SubjectiveNorms(component.Component):
  """This component generates a list of candidate people ."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      components: list[component.Component],
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    """Initializes the BehavioralChoices component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      agent_name: Name of the agent.
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
    self._components = components
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = name
    self._list = [] 

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state
  
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

    self._json = self._components[0].jsonify()
    self._behaviours = [behaviour["behaviour"] for behaviour in self._json]

    nl = "\n"
    
    question = (
        f"Instructions: \n"
        f"Below is a set out of potential behaviours that {self._agent_name} can take in response to the situation. "
        f"For each behaviour, provide a list of 5 people who might have opinions about whether {self._agent_name} "
        f"should do this behaviour or not. Do not include {self._agent_name} in this list. "
        f"For each behaviour, indicate who the person is that is having the opinion, "
        f"and give a rating from -10 to 10 whether they approve or disapprove of the behaviour. "
        f"-10 is the most disapproval,  and 10 is the most approval."
        f"List just the 5 people who have opinions on it, and the rating of the person, for each behaviour. "
        f"Behaviours:\n"
        f"{nl.join(self._behaviours)}"
        f"\n"
        f"Do not include anything other than the list of behaviours and the list of people who might have an opinion. "
        f"Do not include a question or an answer. "
        f"After listing each person, give their approval in the following format: (Approval: number).\n"
    )

    self._state = prompt.open_question(
      question,
      max_characters=5000,
      max_tokens=3000,
      terminators=None
    )

    self._last_chain = prompt

    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

  def jsonify(self) -> None:


    list_1 = re.split(
      r'\n\*\*',
      self._state.replace('**Behaviours:**\n', '')
    )
    list_1 = [item for item in list_1 if item]

    behav_people = []

    for people in list_1:

      list_2 = re.split(
        r'\n',
        people
      )


      pattern_1 = re.compile(r'^\d\.\s')

      pattern_2  = re.compile(r'^-\s')

      pattern_3 = re.compile(r'\*\*$')

      list_3 = [item for item in list_2 if (pattern_1.match(item) or pattern_2.match(item)) and not pattern_3.search(item)]
      
      if len(list_3) > 0:
        behav_people.append(list_3)
        
    self._all_people = []

    for item in range(len(self._json)):

      self._json[item]["people"] = []
      
      for person in behav_people[item]:

        person_name = re.sub(
          r'\d\.\s|-\s',
          '',
          re.search(
            r'(.*)(?=\()',
            person
          ).group(1)
        ).strip()
        person_approval = int(re.search(
          r'(?<=Approval:\s)(-?\d+)(?=\))',
          person
        ).group(1))

        self._json[item]["people"].append({
          "person": person_name,
          "approval": person_approval
        })
        
        if person_name not in self._all_people:
          self._all_people.append(person_name)

    
    print(self._json)
    print(self._all_people)

class PersonMotivations(component.Component):

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      player_config: formative_memories.AgentConfig,
      components: list[component.Component],
      clock_now: Callable[[], datetime.datetime] | None = None,
      num_memories_to_retrieve: int = 100,
      verbose: bool = False,
  ):
    """Initializes the BehavioralChoices component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      agent_name: Name of the agent.
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
    self._components = components
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = name
    self._list = [] 

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state
  
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

    self._json = self._components[0]._json
    self._all_people = self._components[0]._all_people
    nl = "\n"

    question = (
    f"instructions: \n"
    f"below is a set of people that may be approve of disapprove of {self._agent_name}'s actions. "
    f"for each person, indicate the degree to which {self._agent_name} is motivated to take into consideratiom their approval or disapproval. "
    f"in other words, how much will {self._agent_name} take their approval or disapproval into consideration in her actions. "
    f"provide a value from 0 to 100, with 0 being not at all, and 100 being the most infuential, "
    f"in the format (Motivation: number). Remember to use the full scale from 0 to 100 in the ratings. "
    f"Only provide the name of each person and the degree to which {self._agent_name} "
    f"is motivated to take their approval or disapproval into consideration in general. "
    f"People:\n"
    f"{nl.join(self._all_people)}"
    f"Do not provide any explanation or description."
    )

    self._state = prompt.open_question(
      question,
      max_characters=5000,
      max_tokens=3000,
      terminators=None
    )

    self._last_chain = prompt

    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

  def jsonify(self) -> list[dict]:

    list_0 = re.split(r'\n\n', self._state)

    json_output = self._json

    pattern = re.compile(r':\s\d+')

    list_1 = [item for item in list_0 if pattern.search(item)]

    for item in list_1:

      person_motivation = int(re.search(
        r'(?<=:\s)(\d+)',
        item
      ).group(1))

      person_name = re.match(
        r'([^:]*)',
        item
      ).group(1).replace('*','').strip()

      for b in range(len(json_output)):
        for p in range(len(json_output[b]["people"])):
          if json_output[b]["people"][p]["person"] == person_name:
            json_output[b]["people"][p]["motivation"] = person_motivation
        

    return json_output









    

