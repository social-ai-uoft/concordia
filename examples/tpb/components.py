# Author: Rebekah Gelpi
# Concordia Cognitive Model: Theory of Planned Behaviour

"""Agent component for self perception."""
import datetime
import re
import concurrent.futures
import os
import termcolor
import numpy as np

from retry import retry
from scipy.stats import zscore
from typing import Any, Callable, Sequence

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component

from examples.tpb import agent as tpb_agent
from examples.tpb import memory as tpb_memory
from examples.tpb import utils


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
    or a multi-threaded executor.
    
    Args:
      inputs: A list of question texts to pose.
      func: the question function (i.e., the question-asking prompt)"""
    # Single-threaded: run a for loop
    if not self._kwargs['threading']:
      prompts = []
      outputs = []
      for i in inputs:
        prompt, output = func(i)
        prompts.append(prompt)
        outputs.append(output)
    # If threading is enabled, use the thread pool executor.
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
      statements: dict[str, Any] = None,
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

    # Default prompt statements unless they are specified by the input.
    statements = {
      "Memories": "\n" + mems,
      "Traits": self._config.traits,
      "Goals": self._config.goal,
      "Current time": self._clock_now()
    } if statements is None else statements

    self.prompt_builder(prompt, statements)

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
      config: tpb_agent.AgentConfig,
      num_memories: int = 100,
      num_behavs: int = 5,
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
    Initializes the Attitude component.
    
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

    q = (
          f"Instructions: \n"
          f"Given the memories above, and the candidate behaviour below: \n\n"
          "{behav}\n\n"
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
    
    behavs = [item['behaviour'] for item in self._components[0].json()]

    prompts, outputs = self.prompt_batch([q.format(behav=behav) for behav in behavs], self.question)

    self._last_chain = prompts[-1]
    self._state = "\n\n####\n\n".join(outputs)

    self.jsonify()

    assert([] not in [behav['consequences'] for behav in self.json()]), self._err("Did not successfully parse all consequences.")

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

  def jsonify(self) -> None:
    
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
              r'(.*?)(?=:|\(|\s-|\n)',
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

    self._json = output

  @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def _update(self) -> None:

    # Behaviour will be filled in later by the formatter
    q = (
        f"Instructions: \n"
        f"Given the memories above, and the candidate behaviour below: \n\n"
        "{behav}\n\n"
        f"Provide a numbered list of {self._num_people} people who might have opinions about whether "
        f"{self._agent_name} should do this behaviour or not. Do not include {self._agent_name} in "
        f"this list. For each person, include a rating from -10 to 10 indicating whether they approve "
        f"or disapprove of the behaviour. -10 is the most disapproval, and 10 is the most approval.\n"
        f"Do not provide any additional explanation or description.\n"
        f"After listing each person, give their approval in the following format: (Approval: number).\n"
        f"Here is an example: Dave (Approval: 2)."
    )

    behavs = [item['behaviour'] for item in self._components[0].json()]

    prompts, outputs = self.prompt_batch([q.format(behav=behav) for behav in behavs], self.question)
      
    self._last_chain = prompts[-1]
    self._state = "\n\n####\n\n".join(outputs)

    self._json = self.jsonify()

    assert([] not in [behav['people'] for behav in self.json()]), self._err("Did not successfully parse all people.")

class Motivation(TPBComponent):
  """This gets the motivation of a character to take others' opinions into account."""

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
    self._all_people: list[str] = []

  def jsonify(self) -> None:

    motiv_list = self._state.split("\n\n####\n\n")
    self._state = self._state.replace("\n\n####\n\n", "")
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
    
    self._json = output

  @retry(AssertionError, tries = MAX_JSONIFY_ATTEMPTS)
  def _update(self) -> None:

    # Person will be filled in later by the formatter.
    q = (
        f"Instructions: \n"
        f"Given the memories above, and the person below: \n\n"
        "{person}\n\n"
        f"Indicate how motivated {self._agent_name} is "
        "to take into consideration {person}'s approval or disapproval. "
        f"In other words, how much will {self._agent_name} "
        "consider whether {person} approves or disapproves "
        f"of a behaviour when {self._agent_name} is considering how to act in a given situation."
        f"Provide a value from 0 to 100, with 0 being not at all, and 100 being the most infuential, "
        f"in the format (Motivation: number). Remember to consider the full scale from 0 to 100 in the ratings. "
        f"Do not provide any explanation or description."
    ) 

    # This horrible expression generates a list of uniquely identified people across all behaviours.
    self._all_people = list(set.union(*map(set, ([j['person'] for j in i] for i in [item['people'] for item in self._components[0].json()]))))

    prompts, outputs = self.prompt_batch([q.format(person=person) for person in self._all_people], self.question)
      
    self._last_chain = prompts[-1]
    self._state = "\n\n####\n\n".join(outputs)

    self.jsonify()

    assert(len(self._json) > 0), "Did not successfully parse the array into a list of outputs."    

class SubjectiveNorm(TPBComponent):
  """This computes the subjective norms applied to a behaviour."""

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

  def eval_norm(self, people: list[dict]) -> int:
    """Generate a global subjective norm for a behaviour given a list of people.
    
    Args:
      people: A list of dictionaries listing the consequence description, value, and likelihood."""
    vs = [x["approval"] for x in people]
    ls = [x["motivation"] for x in people]
    return(sum([v * l for v, l in zip(vs, ls)]))
          
  def update(self) -> None:
    
    # Take the motivation json
    output = self._components[0].json()
    for i in range(len(output)):
      output[i]['norm'] = self.eval_norm(output[i]['people'])

    self._json = output

class BehaviouralControl(TPBComponent):
  """Behavioural control component."""

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
    Initializes the BehaviouralControl component.
    
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

  def jsonify(self) -> None:
    """Returns the state of the component as a JSON-serializable dictionary."""

    control_list = self._state.split("\n\n####\n\n")
    self._state = self._state.replace("\n\n####\n\n", "")

    # Create a copy of the input json
    output = self._components[0].json()

    for control, behav in zip(control_list, output):
      behav["control"] = int(re.search(
        r'(?<=Probability: )(\d+)',
        control
      ).group(1)) / 100
    
    self._json = output

  def _update(self) -> None:

    # Behaviour will be filled in later by the formatter
    q = (
          f"Instructions: \n"
          f"Given the memories above, and the candidate behaviour below: \n\n"
          "{behav}\n\n"
          f"Provide the probability that {self._agent_name} perceives that {self._config.pronoun()} "
          f"could succeed in doing this behaviour. Indicate with a number from 0 to 100 {self._config.pronoun()}'s "
          f"perception of the probability of succeeding. 0 is impossible, 100 is certain. "
          f"This is not the likelihood that {self._agent_name} will do the behaviour, but rather the likelihood "
          f"that {self._agent_name} would succeed if {self._config.pronoun()} were to do the behaviour. "
          f"Provide the response in the format (Probability: number). "
          f"Remember to consider the full scale from 0 to 100 in the ratings. "
          f"Do not provide any explanation or description."
      )

    behavs = [item['behaviour'] for item in self._components[0].json()]

    prompts, outputs = self.prompt_batch([q.format(behav=behav) for behav in behavs], self.question)
      
    self._last_chain = prompts[-1]
    self._state = "\n\n####\n\n".join(outputs)

    self.jsonify()

    assert(None not in [behav['control'] for behav in self.json()]), self._err("Did not successfully parse all people.")

class BehaviouralIntention(TPBComponent):
  """Synthesizes attitude, norm, and behavioural control into a single behavioural intention."""

  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      config: tpb_memory.TPBAgentConfig,
      components: Sequence[component.Component],
      w: float = 0.5,
      tau: float = 1.,
      num_memories: int = 100,
      clock_now: Callable[[], datetime.datetime] | None = None,
      verbose: bool = False,
      **kwargs
   ):
    """
    Initializes the BehaviouralIntention component.
    
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
    self._w = w
    self._softmax = lambda x : utils.softmax(x, tau)
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
    """Collate the outcomes from a component for each behaviour..
    
    Args:
      measure (str): The component to get the measure from.
      
    Return:
      list: A list of the outputs from the component for each behaviour."""
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
    control = self.collate("control")

    # Weigh the two values
    behavioural_intentions = (self._w * attitudes) + ((1 - self._w) * norms)

    # Compute softmax
    behav_probs = self._softmax(behavioural_intentions)

    # Reweight by the probability of success
    behav_probs = behav_probs * control

    #Normalize
    behav_probs = utils.normalize(behav_probs)

    return behav_probs
  
  def stringify(self) -> str:
    """Return a string containing behaviour, consequences, and subjective norms for each behaviour."""

    output = ""

    for i in range(len(self._components["attitude"].json())):
      behaviour = self._components["attitude"].json()[i]["behaviour"]
      control = self._components["control"].json()[i]["control"]
      consequences = [c["description"] for c in self._components["attitude"].json()[i]["consequences"]]
      values = [c["value"] for c in self._components["attitude"].json()[i]["consequences"]]
      likelihoods = [c["likelihood"] for c in self._components["attitude"].json()[i]["consequences"]]
      people = [c["person"] for c in self._components["norm"].json()[i]["people"]]
      approvals = [c["approval"] for c in self._components["norm"].json()[i]["people"]]
      motivations = [c["motivation"] for c in self._components["norm"].json()[i]["people"]]

      output += f"Possible Behaviour: {behaviour}\n\n"
      output += f"Probability of succeeding at behaviour: {control * 100}%\n\n"
      output += f"Potential Consequences:\n"
      output += f"\n".join([
        f"{i+1}. {consequences[i]} (Value: {values[i]}, Likelihood: {likelihoods[i] * 100}%)" for i in range(len(consequences))
      ])
      output += f"\n\n"
      output += f"Others:\n"
      output += f"\n".join([
        f"{i+1}. {people[i]} (Approval of behaviour: {approvals[i]}, {self._agent_name}'s motivation to consider their views: {motivations[i] * 100}%)" for i in range(len(people))
      ])
      output += "\n\n"

    return output
  
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
  
  def plot(self, file_path: str | os.PathLike | None = None) -> None:
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

    plt.barh(b3, self._softmax(attitudes), height = bw, color = 'darkgreen', label = 'Attitudes')
    plt.barh(b2, self._softmax(norms), height = bw, color = 'limegreen', label = 'Subjective Norms')
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
      self._update()
      if self._verbose:
        self._print("Behavioural Intentions:\n")
        self._print("\n".join([f"{i+1}. {self.collate('behaviour')[i]} (Probability: {round(self.evaluate_intentions()[i], 2)})" for i in range(len(self.collate("behaviour")))]))
    else:
      if self._verbose:
        self._print(f"{self.name()} initialized.")
      self._is_initialized = True
      self._components["thin_goal"].update()
      self._state = self._components["thin_goal"].state()

  def _update(self) -> None:

    behavs = self.collate("behaviour")
    behavioural_intentions = self.evaluate_intentions()

    chosen_behav = np.random.choice(behavs, p=behavioural_intentions)

    self._state = (
      f'After considering {self._config.pronoun(case = "genitive")} options, '
      f"{self._agent_name}'s current goal is to successfully accomplish or complete the following behaviour: {chosen_behav}."
    )

class ThinGoal(TPBComponent):
  """Computes a goal on first activation when no state exists."""
  def __init__(
      self,
      name: str,
      model: language_model.LanguageModel,
      config: tpb_memory.TPBAgentConfig,
      num_memories: int = 100,
      clock_now: Callable[[], datetime.datetime] | None = None,
      verbose: bool = False,
      **kwargs
   ):
    """
    Initializes the ThinGoal component.
    
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

  def _update(self):

    question = (
        f"Instructions: \n"
        f"Given the memories above, restate {self._agent_name}'s goal in one sentence."
    )

    prompt, self._state = self.question(
      question,
    )

    self._last_chain = prompt
    
class TPBModel(component.Component):
  """Full sequential component architecture for the Theory of Planned Behaviour."""

  def __init__(
      self,
      name: str,
      components: Sequence[TPBComponent],
      verbose: bool = False
  ):
    
    self._name = name
    self._state = ''
    self._components: dict[str, TPBComponent] = {}
    for component in components:
      self._components[component.name()] = component

  def name(self) -> str:
    return self._name

  def state(self) -> str:
    return self._state

  def observe(self, observation: str) -> None:
    self._components["memory"].observe(observation)

  def component(self, name) -> component.Component:
    return self._components[name]

  def update(self) -> None:
    # First, the TPB components...
    self._components["behaviour"].update()
    self._components["attitude"].update()
    self._components["people"].update()
    self._components["motivation"].update()
    self._components["norm"].update()
    self._components["control"].update()
    self._components["intention"].update()

    # Store the deliberation summary from all of the TPB components...
    deliberation = self._components["memory"].summarize(
      self._components["intention"].stringify()
    )
    # and then put them into the working memory as the deliberation component
    self._components["memory"].observe(deliberation, wm_loc = "delib")

    # After deliberations are complete, synthesize the TPB model into the plan
    self._components["situation"].update()
    self._components["plan"].update()
    self._state = self._components["plan"].state()

    plan = self._components["memory"].summarize(
      self._state, kind = "plan"
    )
    
    # Add the plan as the action component of the working memory
    self._components["memory"].observe(plan, wm_loc = "action")