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


class QuizMetric(component.Component):
    """How well does the agent do on a multiple choice exam?"""

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
        self._personality_scores = []
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
        num_correct = 0
        conscientiousness = []
        agreeableness = []
        emotionalstability = []
        openness = []
        extraversion = []


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
                    f"Question: {question['question']}\n{self._player_name}'s answer: {question['options'][agent_answer]}\nCorrect answer: {question['options'][question['correct_answer']]}\n"
                )
                print(agent_answer)

            # SARAH XU: 'question' will give the whole question, inclusing options, question, and correct answer
                # if we only want the question, replace question with question['question']
            self._results.append([self._player_name, 1000, question['question'], agent_answer, question["correct_answer"]])

            if agent_answer == question["correct_answer"]:
                num_correct += 1

        # SARAH XU: Added a TIPI Scoring logic
            if question_num == 1:
                extraversion.append(agent_answer + 1)
            elif question_num == 2:
                agreeableness.append(agent_answer + 1)
            elif question_num == 3:
                conscientiousness.append(agent_answer + 1)
            elif question_num == 4:
                emotionalstability.append(agent_answer + 1)
            elif question_num == 5:
                openness.append(agent_answer + 1)
            elif question_num == 6:
                extraversion.append(agent_answer + 1)
            elif question_num == 7:
                agreeableness.append(agent_answer + 1)
            elif question_num == 8:
                conscientiousness.append(agent_answer + 1)
            elif question_num == 9:
                emotionalstability.append(agent_answer + 1)
            elif question_num == 10:
                openness.append(agent_answer + 1)
            
        conscientiousness_score = conscientiousness[0] + (8 - conscientiousness[1]) / 2
        agreeableness_score = agreeableness[1] + (8 - agreeableness[0]) / 2
        emotionalstability_score = emotionalstability[1] + (8 - emotionalstability[0]) / 2
        openness_score = openness[0] + (8 - openness[1]) / 2
        extraversion_score = extraversion[0] + (8 - extraversion[1]) / 2

        self._personality_scores = [conscientiousness_score,
                                   agreeableness_score,
                                   emotionalstability_score,
                                   openness_score,
                                   extraversion_score]
        print(f"Conscientiousness Score: {conscientiousness_score}")
        print(f"Agreeableness Score: {agreeableness_score}")
        print(f"Emotional Stability Score: {emotionalstability_score}")
        print(f"Openness to Experience Score: {openness_score}")
        print(f"Extraversion Score: {extraversion_score}")

        answer_str = (
            f"Agent scored {num_correct}/{len(self._exam['questions'])} on quiz."
        )

        datum = {
            "time_str": self._clock.now().strftime("%H:%M:%S"),
            "clock_step": self._clock.get_step(),
            "timestep": self._timestep,
            "value_float": num_correct,
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
    
    def get_personality_scores(
        self,
    ) -> list:
        return self._personality_scores

    def state(
        self,
    ) -> str | None:
        """Returns the current state of the component."""
        return ""
