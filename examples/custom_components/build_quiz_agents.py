from __future__ import annotations

from concordia.associative_memory import formative_memories
from concordia.associative_memory import blank_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.language_model import language_model
from concordia.document import interactive_document

import re
import random
import datetime
import pickle
from typing import Sequence



# Three scenarios being tested
# 1. People from 1924 and 2024 are taking a pop culture trivia quiz.
# 2. People with high and low levels of medical knowledge are taking an MCAT preparatory quiz.
# 3. People with different scores on the Big Five factors 
# (Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism)
# are taking a personality quiz to measure their Big Five score.

def trait_level() -> str:
    return random.choice(['very high', 'somewhat high', 'medium', 'somewhat low', 'very low'])

def build_clock(year: int) -> game_clock.MultiIntervalClock:
    #@title Make the clock
    time_step = datetime.timedelta(minutes=20)
    SETUP_TIME = datetime.datetime(hour=20, year=year, month=10, day=1)

    START_TIME = datetime.datetime(hour=18, year=year, month=10, day=2)
    clock = game_clock.MultiIntervalClock(
        start=SETUP_TIME,
        step_sizes=[time_step, datetime.timedelta(seconds=10)])
    return clock

def generate_names(model: language_model.LanguageModel, gender: str, country: str = None) -> list[str]:

    """
    Generate a list of names matching the gender provided.
    """

    if gender == 'male':
        g = 'men'
    else:
        g = 'women'

    question = f'Please provide a list of 50 first names for {gender} in the form of a comma-separated list. '

    if country is not None:
        question += f'The names should be names typical of {country}.'
    
    attempts = 0
    names = '\n\n\n\n'
    while ',' not in names.split('\n')[2]:
        attempts += 1
        if attempts > 4:
            raise language_model.InvalidResponseError(question)
        names = model.sample_text(
            prompt=question,
            max_tokens=2500,
            max_characters=5000
        )
        

    return names.split('\n')[2].replace('.', '').split(',')

class QuizAgent():
    """Agent for completing trivia quizzes."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        name: str,
        agent_id: int,
        age: int,
        gender: str,
        traits: str,
        context: str,
        clock: game_clock.MultiIntervalClock,
        backstory: str = None
    ):
        
        self._model = model
        self._clock = clock
        self.name = name
        self.agent_id = agent_id
        self.age = age
        self.gender = gender
        self.traits = traits
        self.context = context
        self.backstory = self.add_backstory() if backstory is None else backstory

    def add_backstory(
            self
        ) -> str:
        """Create a brief summary of an agent given the relevant characteristics and context.

        Args:
            name: The name of the agent.
            gender: The gender of the agent.
            traits: The traits of the agent.
            context: Relevant context for the agent summary.

        Returns:
            Descriptive text about the agent.
        """

        prompt = interactive_document.InteractiveDocument(self._model)

        if self.context:
            prompt.statement(self.context)

        question = (
            f'Given the following traits:\n{self.traits}'
            f'\n create a short summary about a {self.gender} character called {self.name}.'
            ' Write a summary of the person:'
            ' what their job is, what a typical day is is like, what are their'
            ' goals, desires, hopes, dreams, and aspirations. Also write about'
            ' their duties, responsibilities, and obligations. Make sure to include'
            ' information about the traits listed.'
        )
        if self.context:
            question += f'Take into account the following context: {self.context}'
        result = prompt.open_question(
            question,
            max_characters=2500,
            max_tokens=2500,
            terminators=[],
        )
        result = re.sub(r'\.\s', '.\n', result)

        query = '\n'.join([
            (
                'Replace all the pronouns in the following text with the name'
                f' {self.name}.'
            ),
            'The text:',
            result,
        ])

        description = self._model.sample_text(query)
        description = re.sub(r'\.\s', '.\n', description)

        return description

    def export_agent(
        self
    ) -> dict:
        return {
            'name': self.name,
            'agent_id': self.agent_id,
            'age': self.age,
            'gender': self.gender,
            'traits': self.traits,
            'context': self.context,
            'backstory': self.backstory
        }
    
    def save_agent(
        self,
        location
    ) -> None:
        """
        Save the agent to the specified file location.
        """
        with open(location, 'wb') as file:
            pickle.dump(self.export_agent(), file, protocol=pickle.HIGHEST_PROTOCOL)
    
    @classmethod
    def import_agent(
        cls,
        model,
        clock,
        agent_dict
    ) -> QuizAgent:
        """
        Load a QuizAgent object using a dictionary of agent details.
        """
        return QuizAgent(
            model,
            agent_dict['name'],
            agent_dict['agent_id'],
            agent_dict['age'],
            agent_dict['gender'],
            agent_dict['traits'],
            agent_dict['context'],
            clock,
            agent_dict['backstory']
        )
    
    @classmethod
    def load_agent(
        cls,
        model,
        clock,
        location
    ) -> QuizAgent:
        """
        Load a QuizAgent object from a specified pickle location.
        """
        with open(location, 'rb') as file:
            agent_dict = pickle.load(file)
        
        return cls.import_agent(model, clock, agent_dict)
            
        

def build_random_agent(
        model: language_model.LanguageModel,
        names: Sequence[Sequence[str]],
        clock: game_clock.MultiIntervalClock,
        scenario: str,
        agent_id: int = None,
        seed: int = None
) -> str:
    
    if seed is not None:
        seed = random.seed(seed)

    current_year = clock.now().year
    age = 18 + random.randint(0,46)
    gender = random.choice(['male', 'female'])
    list = 0 if gender == 'male' else 1
    agent_id = random.randint(0,len(names[list])) if agent_id is None else agent_id
    name = names[list][agent_id].strip()

    if scenario == 'trivia':

        traits = 'general knowledge: ' + random.choice(['very high', 'high', 'medium', 'low', 'very low'])

        context = (
            f'{name} was born in {current_year - age}. The year is now {current_year}, and {name} is now {age} years old. '
            + f'{name} is about to take a trivia quiz for fun.'
        )
        

    if scenario == "mcat":

        gen_k = trait_level()
        med_k = trait_level()

        traits = (
            'general knowledge: ' + gen_k + 
            '; medical knowledge: ' + med_k 
        )

        context = (
            f'{name} was born in {current_year - age}. The year is now {current_year}, and {name} is now {age} years old. ' +
            f'{name} has {med_k} medical knowledge relative to the average person in the year {current_year}.' +
            f'{name} is taking a quiz including questions from the MCAT for fun.'
        )

    if scenario == "personality":

        o_level = trait_level()
        c_level = trait_level()
        e_level = trait_level()
        a_level = trait_level()
        n_level = trait_level()

        traits = (
            'openness: ' + o_level +
            '; conscientiousness: ' + c_level +
            '; extraversion: ' + e_level +
            '; agreeableness: ' + a_level +
            '; neuroticism: ' + n_level
        )

        context = (
            f'{name} was born in {current_year - age}. The year is now {current_year}, and {name} is now {age} years old. ' +
            f'In terms of personality, {name} has a {o_level} level of openness, ' +
            f'a {c_level} level of conscientiousness, a {e_level} level of extraversion, ' +
            f'a {a_level} level of agreeableness, and a {n_level} level of neuroticism. ' +
            f'{name} is taking a personality quiz for fun.'
        )

    print(f'''
Creating backstory for {name} using the following information:
Age: {age}
Current year: {current_year}
Traits: {traits}
Gender: {gender}
''')
        
    agent = QuizAgent(model, name, agent_id, age, gender, traits, context, clock)

    return agent





        