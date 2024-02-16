"""Give an agent a multiple choice question."""

from collections.abc import Sequence
import json

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import component
from concordia.utils import measurements as measurements_lib

import torch

# DEFAULT_SCALE = ('abhorrent', 'wrong', 'neutral', 'right', 'praiseworthy')
DEFAULT_CHANNEL_NAME = "multiple_choice_quiz"


class LogitsQueryMetric(component.Component):
    """Output the logits of specific words in the model's response."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        player_name: str,
        clock: game_clock.GameClock,
        exam_json_path: str,
        name: str = "LogitQueryScores",
        verbose: bool = False,
        measurements: measurements_lib.Measurements | None = None,
        channel: str = DEFAULT_CHANNEL_NAME,
        query=None,
    ):
        """Initializes the metric.

        Args:
          model: The language model to use.
          player_name: The player to ask about.
          clock: The clock to use.
          name: The name of the metric.
          scale: The scale of the metric, uses default if None.
          verbose: Whether to print the metric.
          measurements: The measurements to use.
          channel: The name of the channel to push data
        """
        self._model = model
        assert self._model._logits, "Model must be set to logit mode to use this metric."

        self._name = name
        self._clock = clock
        self._verbose = verbose
        self._player_name = player_name
        # self._scale = scale
        self._measurements = measurements
        self._channel = channel

        self._timestep = 0
        self._results = [] #[name, agent_id, question, answer]
        # Load the exam
        with open(exam_json_path, "r") as f:
            self._exam = json.load(f)
    
        assert query is not None, "Query must be provided"
        self._query = []
        for q in query:
            self._query.append((q, self._model._tokenizer.encode(q)[0]))

    def name(
        self,
    ) -> str:
        """See base class."""
        return self._name

    def observe(self, observation: str) -> None:
        """See base class."""
        for i in range(len(self._exam["questions"])):
            question = self._exam["questions"][i]

            agent_answer = self._model.sample_text(
                prompt=f"{observation}\n{question['question']}",
            )
            next_token_probs = torch.softmax(agent_answer, -1)

            probs = {}
            for word, token in self._query:
                probs[word] = next_token_probs[token].item()

            # Normalize the probabilities
            total = sum(probs.values())
            for word in probs:
                probs[word] = probs[word] / total

            if self._verbose:
                print(
                    f"To the question {question['question']}, the agent answered\n {probs}"
                )
            
            self._results.append([self._player_name, 1000, question['question'], probs, question["correct_answer"]])

        answer_str = (
            f"Agent completed the quiz."
        )

        datum = {
            "time_str": self._clock.now().strftime("%H:%M:%S"),
            "clock_step": self._clock.get_step(),
            "timestep": self._timestep,
            # "value_float": num_correct,
            "value_str": answer_str,
            "player": self._player_name,
        }
        if self._measurements:
            self._measurements.publish_datum(self._channel, datum)

        datum["time"] = self._clock.now()

        if self._verbose:
            print(f"{self._name} of {self._player_name}: {answer_str}")
        self._timestep += 1

    def get_results(
        self,
    ) -> list:
        return self._results

    def state(
        self,
    ) -> str | None:
        """Returns the current state of the component."""
        return ""
