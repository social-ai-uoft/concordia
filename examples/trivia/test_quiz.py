import importlib

# Logging
import logging
logging.basicConfig(level=logging.ERROR, filename='test_quiz.log')
logger = logging.getLogger('ollama')

# Setup sentence encoder
from sentence_transformers import SentenceTransformer
st5_model = SentenceTransformer('sentence-transformers/sentence-t5-base')
embedder = st5_model.encode

# Set up LLM
import sys
import os
path = os.path.abspath('../..')
if path not in sys.path:
    sys.path.insert(0, path)
path = os.path.abspath('..')
if path not in sys.path:
    sys.path.insert(0, path)

from concordia.language_model import ollama_model
model = ollama_model.OllamaLanguageModel(
    model_name='llama2:13b'
    # model_name='mixtral'
)

import custom_components.build_quiz_agents as build_agents
importlib.reload(build_agents)

m_names = build_agents.generate_names(model, 'men')
f_names = build_agents.generate_names(model, 'women')
names = [m_names, f_names]

print(len(names[0]), len(names[1]))

for i in range(min(len(names[0]), len(names[1]))):
    scenario = "trivia"
    clock = build_agents.build_clock(2024)
    agent = build_agents.build_random_agent(model, names, clock, scenario, i)

    filepath = f"./agents/2024/trivia/{agent.agent_id}.pkl"
    
    agent.save_agent(filepath)

from concordia.utils import measurements as measurements_lib
measurements = measurements_lib.Measurements()
import examples.custom_components.quiz_metric as qm
importlib.reload(qm)

all_results = []
run_number = []
N_RUNS = 5

for i in range(min(len(names[0]), len(names[1]))):

    for j in range(N_RUNS):

        agent = build_agents.QuizAgent.load_agent(model, clock, f'./agents/2024/trivia/{i}.pkl')

        if scenario == 'personality':
            test_context = (
                '\n'
                f'{agent.name} is taking a personality quiz just for fun.'
                f'Here are a number of personality traits that may or may not apply to {agent.name}. '
                f'Please indicate the extent to which {agent.name} would agree or disagree with that statement. '
                f'{agent.name} should rate the extent to which the pair of traits applies to '
                f'{"her" if agent.gender == "female" else "him"}, even if one characteristic applies more strongly than the other.'
                '\n'
            )
            file_path = './quizzes/big_five_questions.json'
        if scenario == 'trivia':
            test_context = (
                '\n'
                f"{agent.name} is taking a pop culture trivia quiz just for fun."
                f"If {agent.name} doesn't know the answer, {'she' if agent.gender == 'female' else 'he'} will still guess "
                f"and begin {'her' if agent.gender == 'female' else 'his'} answer with a single choice. "
                f"How would {agent.name} answer the following question?"
                '\n'
            )
            file_path = './quizzes/trivia_questions.json'
        if scenario == "mcat":
            test_context = (
                '\n'
                f"{agent.name} is taking an MCAT preparatory quiz just for fun."
                f"If {agent.name} doesn't know the answer, {'she' if agent.gender == 'female' else 'he'} will still guess "
                f"and begin {'her' if agent.gender == 'female' else 'his'} answer with a single choice. "
                f"How would {agent.name} answer the following question?"
                '\n'
            )
            file_path = './quizzes/mcat_questions.json'

        context = agent.backstory + test_context

        quiz_metric = qm.QuizMetric(model, agent, file_path, measurements = measurements, verbose = True)
        results = quiz_metric.observe(context, return_data=True)

        run_number.append(j)
        all_results.append(results)

import csv

with open('./data/llama2_13b_trivia_2024_agents.csv', 'w') as newfile:
    writer = csv.writer(newfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['agent_id', 'agent_name', 'traits', 'question', 'answer', 'correct', 'run'])
    for i in range(len(all_results)):
        run = run_number[i]
        results = all_results[i]
        for result in results:
            row = [item.strip() for item in result.split(",")]
            row.append(run)
            writer.writerow(row)