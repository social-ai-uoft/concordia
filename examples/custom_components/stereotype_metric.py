"""Give an agent a multiple choice quiz."""

from collections.abc import Sequence
import json
import datetime

from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.clocks import game_clock
from concordia.typing import component
from concordia.utils import measurements as measurements_lib


# DEFAULT_SCALE = ('abhorrent', 'wrong', 'neutral', 'right', 'praiseworthy')
DEFAULT_CHANNEL_NAME = "multiple_choice_quiz"
#@title Make the clock
time_step = datetime.timedelta(minutes=20)
SETUP_TIME = datetime.datetime(hour=20, year=2024, month=10, day=1)

START_TIME = datetime.datetime(hour=18, year=2024, month=10, day=2)


class QuizMetric(component.Component):
    """How well does the agent do on a multiple choice exam?"""

    def __init__(
        self,
        model: language_model.LanguageModel,
        agent,
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
        self._clock = game_clock.MultiIntervalClock(
            start=SETUP_TIME,
            step_sizes=[time_step, datetime.timedelta(seconds=10)])
        self._verbose = verbose
        self._player_name = agent.name
        self._player_id = agent.agent_id
        self._traits = agent.traits
        # self._scale = scale
        self._measurements = measurements
        self._channel = channel

        self._timestep = 0
        self._results = [] #[name, agent_id, question, answer]
        self._personality_scores = []
        # Load the exam
        with open(exam_json_path, "r") as f:
            self._exam = json.load(f)

    def name(
        self,
    ) -> str:
        """See base class."""
        return self._name

    def observe(self, observation: str, return_data = False) -> None:
        """See base class."""
        score = 0


        # Iterate over the list of questions and ask them one by one

        # for question in self._exam["questions"]:
        for i in range(len(self._exam["questions"])):
            question_num = i + 1
            question = self._exam["questions"][i]
            doc = interactive_document.InteractiveDocument(self._model)
            agent_answer = doc.multiple_choice_question(
                f"{observation}\n{question['question']}", question["options"]
            )

            if self._verbose:
                print(
                    f"Question: {question['question']}\n{self._player_name}'s answer: ({agent_answer}) {question['options'][agent_answer]}\n"
                )

            score += (4 - agent_answer)

        print(f"Score: {score}/40.")

        datum = {
            "time_str": self._clock.now().strftime("%H:%M:%S"),
            "clock_step": self._clock.get_step(),
            "timestep": self._timestep,
            "score": score,
            "player": self._player_name,
        }
        if self._measurements:
            self._measurements.publish_datum(self._channel, datum)

        datum["time"] = self._clock.now()

        # if self._verbose:
        #     print(f"{self._name} of {self._player_name}: {answer_str}")
        self._timestep += 1

        if return_data:
            return self.get_results()


    def get_results(
        self,
    ) -> list:
        return self._results
    
    def get_personality_scores(
        self,
    ) -> list:
        return self._personality_scores

    def state(
        self,
    ) -> str | None:
        """Returns the current state of the component."""
        return ""
