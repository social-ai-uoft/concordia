"""Give an agent a multiple choice quiz."""

from collections.abc import Sequence
import json

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import clock as game_clock
from concordia.typing import component
from concordia.utils import measurements as measurements_lib


# DEFAULT_SCALE = ('abhorrent', 'wrong', 'neutral', 'right', 'praiseworthy')
DEFAULT_CHANNEL_NAME = "multiple_choice_quiz"


class LogitsMetric(component.Component):
    """Output the logits of the model's response."""

    def __init__(
        self,
        model: language_model.LanguageModel,
        player_name: str,
        clock: game_clock.GameClock,
        exam_json_path: str,
        name: str = "QuizScore",
        verbose: bool = False,
        measurements: measurements_lib.Measurements | None = None,
        channel: str = DEFAULT_CHANNEL_NAME,
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
        self._name = name
        self._clock = clock
        self._verbose = verbose
        self._player_name = player_name
        # self._scale = scale
        self._measurements = measurements
        self._channel = channel

        self._timestep = 0
        self._results = [] #[name, agent_id, question, answer]
        self._logits = []
        # Load the exam
        with open(exam_json_path, "r") as f:
            self._exam = json.load(f)

    def name(
        self,
    ) -> str:
        """See base class."""
        return self._name

    def observe(self, observation: str) -> None:
        """See base class."""
        for i in range(len(self._exam["questions"])):
            question = self._exam["questions"][i]

            # If model.logit is true, agent_answer will be logits
            # Otherwise, agent_answer will be the text response
            agent_answer = self._model.sample_text(
                prompt=f"{observation}\n{question['question']}",
            )

            if self._verbose:
                print(
                    f"To the question {question['question']}, the agent answered\n {agent_answer}"
                )
                print(agent_answer)
            
            self._results.append([self._player_name, 1000, question['question'], agent_answer, question["correct_answer"]])

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
    
    def get_logits(
        self,
    ) -> list:
        return self._logits

    def state(
        self,
    ) -> str | None:
        """Returns the current state of the component."""
        return ""
