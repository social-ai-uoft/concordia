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

  def update(self) -> None:
    mems = '\n'.join(
        self._memory.retrieve_recent(
            self._num_memories_to_retrieve, add_time=True
        )
    )

    prompt = interactive_document.InteractiveDocument(self._model)
    prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')

    prompt.statement(f'Traits of {self._agent_name}: {self._traits}')
    prompt.statement(f'Goals of {self._agent_name}: {self._goal}')

    # Compare more or less likely to ONLY the most likely.
    question = (
        f'Given the memories above, list 10 behaviours that {self._agent_name} '
        f'could take in this situation. Include a variety of behaviours, '
        f'including more and less likely behaviours. '
    )

    self._state = prompt.open_question(
        question,
        answer_prefix=f'Ten behaviours that {self._agent_name} could take: \n',
        max_characters=3000,
        max_tokens=1000,
        terminators=None
    )

    self._list = self._model.sample_text(
      f"Reformat this list of behaviours into an array of JSON objects, "
      f"with each object containing an entry called 'behaviour' that "
      f"includes the behaviour and its description.\n"
      f'{self._state}'
    )

    replacements = [
      ('^\n*?.*?\n*?\[', '['),
      (r'\n', '')
    ]
    
    for old, new in replacements:
      self._list = re.sub(old, new, self._list)

    self._last_chain = prompt

    # # Strip leading whitespace, numbers, and remove any "Based on...", and restrict the list to a maximum size of 10 items.
    # self._list = [
    #   re.sub(r'^\s*?\d+\.\s', '', behav) for behav in re.sub(r'\n+', '\n', self._state).split(sep='\n') if not re.match(r'^\s*?Based\son', behav)
    # ][:10]

    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

class BehavioralConsequences(component.Component):
  """This component generates the potential consequences of a behavior."""

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
    """Initializes the BehavioralConsequences component.

    Args:
      name: Name of the component.
      model: Language model.
      memory: Associative memory.
      player_config: Name of the agent.
      components: A sequence of input components.
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
    self._components = components
    self._name = name

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

    for component in self._components:
      self._list = json.loads(component._list)

    self._output = []
    for item in self._list:
      prompt = interactive_document.InteractiveDocument(self._model)
      prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

      if self._clock_now is not None:
        prompt.statement(f'Current time: {self._clock_now()}.\n')

      prompt.statement(
          f'Instructions: \n'
          f'Given the memories above, evaluate the potential consequences '
          f'of {self._agent_name} engaging in the following behaviours. '
          f'Give five potential consequences, and include a variety of '
          f'positive or negative potential consequences.\n'
          f'Behaviours: \n'
      )
      question = (
        ' - '.join([value for value in item.values() if value])
      )
      self._state = prompt.open_question(
        question,
        max_characters=1000,
        max_tokens=1000,
        terminators=None
      )
      self._output.append(self._state)
    
    self._consequences = []
    for item in self._output:
      _item = self._model.sample_text(
        f"Convert the following list of consequences into an array of JSON objects with "
        f"each object containing an entry termed 'consequence' including the consequence and "
        f"an entry termed 'description' including the description of the consequence:\n"
        f"{item}",
        terminators=None
      )

      replacements = [
        ('^\n*?.*?\n*?\[', '['),
        (r'\n', '')
      ]
      
      for old, new in replacements:
        _item = re.sub(old, new, _item)
      
      self._consequences.append(_item)
      

    # question = (
    #   f'Instructions: \n'
    #   f'Given the memories above, evaluate the potential consequences '
    #   f'of {self._agent_name} engaging in the following behaviours. '
    #   f'Give five potential consequences, and include a variety of '
    #   f'positive or negative potential consequences.\n'
    #   f"Behaviours: \n"
    #   f"\n".join(
    #     [f"{str(x[0])}. {' - '.join(x[1].values())}" for x in zip([i for i in range(len(self._list))], self._list)]
    #   )
    # )

    # self._state = prompt.open_question(
    #   question,
    #   max_characters=3000,
    #   max_tokens=1000,
    #   terminators=None
    # )

    # self._list = self._model.sample_text(
    #   f"Reformat this list of behaviours and consequences into an array of JSON objects, "
    #   f"with each object containing an entry called 'behaviour' that includes the "
    #   f"the behaviour and its description, and an entry called 'consequences' that "
    #   f"includes an array of the potential consequences."
    #   f'{self._state}'
    # )


    self._last_chain = prompt
    if self._verbose:
      print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')



class BehavioralEvaluation(component.Component):
  """This component generates an evaluation of a behavior."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      components: Sequence[component.Component] | None,
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
    self._agent_name = agent_name
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = name
    self._components = components
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
    prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')


    # For evaluate. Should evaluate, IF SUCCESSFUL.
    prompt.statement(
        f'Instructions: \n'
        f'Given the memories above, evaluate the following behaviour '
        f'on the extent to which {self._agent_name} would like the likeliest '
        f'outcome of each behaviour, where 1 indicates that {self._agent_name} '
        f'would very much dislike the outcome, and 10 indicates that '
        f'{self._agent_name} would very much like the outcome. '
        f'Provide only the evaluation with a number out of 10, '
        f'in the same order that each behaviour was presented.'
    )


    for component in self._components:
      self._state = prompt.open_question(
        '\n'.join(component._list),
        answer_prefix=f'{self._agent_name} would rate these behaviours as: ',
        max_characters=3000,
        max_tokens=1000,
        terminators=None
      )

      # Generate list
      self._list = re.sub(
        r'^\n?(.+)\n\n', '', self._state
      ).split('\n')

      for i in range(len(self._list)):
        if re.search(r'\d+(?=/10)', self._list[i]) is not None:
          self._list[i] = re.search(r'\d+(?=/10)', self._list[i]).group(0)
        elif re.search(r'\d+(?=\sout)', self._list[i]) is not None:
          self._list[i] = re.search(r'\d+(?=\sout)', self._list[i]).group(0)
        elif re.search(r'(?<=Rating:\s)\d', self._list[i]) is not None:
          self._list[i] = re.search(r'(?<=Rating:\s)\d', self._list[i]).group(0)
        else:
          self._list[i] = self._list[i]

      self._last_chain = prompt
      if self._verbose:
        print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

      # for element in component._list:
      #   self._state = prompt.open_question(
      #     element,
      #     answer_prefix=f'{self._agent_name} would rate this behaviour as: ',
      #     max_characters=3000,
      #     max_tokens=1000,
      #     terminators=('This is because', )
      #   )
      #   self._list.append(self._state)
      
      #   self._last_chain = prompt
      #   if self._verbose:
      #     print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

      # with concurrent.futures.ThreadPoolExecutor() as executor:
      #   self._list = executor.map(lambda x: prompt.open_question(
      #     '\n'.join([question, x]),
      #     answer_prefix=f'{self._agent_name} would rate this behaviour as: ',
      #     max_characters=3000,
      #     max_tokens=1000,
      #     terminators=None
      #   ), component._list)
      #   self._last_chain = prompt
      #   if self._verbose:
      #     print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

class InterpersonalEvaluation(component.Component):
  """This component generates a predicted evaluation of a behavior by another person."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      other_name: str,
      components: Sequence[component.Component] | None,
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
    self._agent_name = agent_name
    self._other_name = other_name
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = name
    self._components = components
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
    prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')


    prompt.statement(
        f'Instructions: \n'
        f'Given the memories above, evaluate the following behaviour '
        f'on the extent to which {self._other_name} would like for '
        f'{self._agent_name} to engage in the behaviour, '
        f'where 1 indicates that {self._other_name} would very much'
        f'dislike for {self._agent_name} to do it, '
        f'and 10 indicates that {self._other_name} would very much'
        f'like for {self._agent_name} to do it. '
        f'Provide only the evaluation with a number out of 10, '
        f'in the same order that each behaviour was presented.'
    )


    for component in self._components:
      self._state = prompt.open_question(
        '\n'.join(component._list),
        answer_prefix=f'{self._other_name} would likely rate these behaviours as: ',
        max_characters=3000,
        max_tokens=1000,
        terminators=None
      )

      # Generate list
      self._list = re.sub(
        r'^\n?(.+)\n\n', '', self._state
      ).split('\n')

      for i in range(len(self._list)):
        if re.search(r'\d+(?=/10)', self._list[i]) is not None:
          self._list[i] = re.search(r'\d+(?=/10)', self._list[i]).group(0)
        elif re.search(r'\d+(?=\sout)', self._list[i]) is not None:
          self._list[i] = re.search(r'\d+(?=\sout)', self._list[i]).group(0)
        elif re.search(r'(?<=Rating:\s)\d', self._list[i]) is not None:
          self._list[i] = re.search(r'(?<=Rating:\s)\d', self._list[i]).group(0)
        else:
          self._list[i] = self._list[i]

      self._last_chain = prompt
      if self._verbose:
        print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')

class BehavioralSuccess(component.Component):
  """This component generates a success evaluation of a behavior."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      memory: associative_memory.AssociativeMemory,
      agent_name: str,
      components: Sequence[component.Component] | None,
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
    self._agent_name = agent_name
    self._clock_now = clock_now
    self._num_memories_to_retrieve = num_memories_to_retrieve
    self._name = name
    self._components = components
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
    prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

    if self._clock_now is not None:
      prompt.statement(f'Current time: {self._clock_now()}.\n')


    prompt.statement(
        f'Instructions: \n'
        f'Given the memories above, evaluate the following behaviour '
        f'on the extent to which {self._agent_name} believes that '
        f'they are likely to be successful in engaging in the behaviour, '
        f'where 1 indicates that {self._agent_name} believes that '
        f'they are very unlikely to succeed, and 10 indicates that '
        f'{self._agent_name} believes that they are very likely to succeed. '
        f'Provide only the evaluation with a number out of 10, '
        f'in the same order that each behaviour was presented.'
    )


    for component in self._components:
      self._state = prompt.open_question(
        '\n'.join(component._list),
        answer_prefix=f'{self._agent_name} would rate these behaviours as: ',
        max_characters=3000,
        max_tokens=1000,
        terminators=None
      )

      # Generate list
      self._list = re.sub(
        r'^\n?(.+)\n+', '', self._state
      ).split('\n')

      for i in range(len(self._list)):
        if re.search(r'\d+(?=/10)', self._list[i]) is not None:
          self._list[i] = re.search(r'\d+(?=/10)', self._list[i]).group(0)
        elif re.search(r'\d+(?=\sout)', self._list[i]) is not None:
          self._list[i] = re.search(r'\d+(?=\sout)', self._list[i]).group(0)
        elif re.search(r'(?<=Rating:\s)\d', self._list[i]) is not None:
          self._list[i] = re.search(r'(?<=Rating:\s)\d', self._list[i]).group(0)
        else:
          self._list[i] = self._list[i]

      self._last_chain = prompt
      if self._verbose:
        print(termcolor.colored(self._last_chain.view().text(), 'green'), end='')


        # Step 1: First get 10 potential behaviours (done)
        # - List 5 potential likely positive and likely negative consequences of the behaviour
        #   - How likely is that consequence, given you successfully try it?
        #   - How good or bad is that consequence, given that it occurs?
        #   - Get the sum of of the product of belief * evaluation
        #   - Something something there are some things where you immediately have the positive or
        #     negative consequences of the action as a single thing, as opposed to having it happen twice
        # Step 2: Subjective norms
        # - Get the relevant potential people who could be impacted for the behaviour
        #   - There might be individuals who might believe that you should or should not perform the behaviour
        #   - Who might approve or disapprove of the behaviour?
        #   - Different consequences: different referents?
        #   - What is {person}'s opinion on you doing {behaviour} from 1 to 10
        #   - How important is this person's approval to you?
        #   - Sum of the product of opinion * importance
        # Step 3: Attitude + sujective norms = intention
        # Step 4: Perceived control
        # - For each behaviour, please list any factors that would make it easy or difficult for you to do this behaviour RIGHT NOW?
        # - Feed this into, AFTER potential behaviour but BEFORE likelihood of consequences.
        #
        # When it softmaxes on outcome, generate a specific set of plan to engage in this behaviour (as a possibility)