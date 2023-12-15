import datetime
from typing import Callable
from typing import Sequence

from concordia.associative_memory import associative_memory
from concordia.document import interactive_document
from concordia.language_model import language_model
from concordia.typing import component
import termcolor


class MoodBySituation(component.Component):
    """What would the mood of the agent be in a situation like this?"""

    def __init__(
            self,
            name: str,
            model: language_model.LanguageModel,
            memory: associative_memory.AssociativeMemory,
            agent_name: str,
            components=Sequence[component.Component] | None,
            clock_now: Callable[[], datetime.datetime] | None = None,
            num_memories_to_retrieve: int = 25,
            verbose: bool = False,
    ):
        """Initializes the MoodBySituation component.

        Args:
            name: The name of the component.
            model: The language model to use.
            memory: The memory to use.
            agent_name: The name of the agent.
            components: The components to condition the answer on.
            clock_now: time callback to use for the state.
            num_memories_to_retrieve: The number of memories to retrieve.
            verbose: Whether to print the state of the component.
        """

        self._verbose = verbose
        self._model = model
        self._memory = memory
        self._state = ''
        self._components = components or []
        self._agent_name = agent_name
        self._clock_now = clock_now
        self._num_memories_to_retrieve = num_memories_to_retrieve
        self._name = name

    def name(self) -> str:
        return self._name

    def state(self) -> str:
        return self._state

    def update(self) -> None:
        prompt = interactive_document.InteractiveDocument(self._model)

        mems = '\n'.join(
                self._memory.retrieve_recent(
                        self._num_memories_to_retrieve, add_time=True
                )
        )

        prompt.statement(f'Memories of {self._agent_name}:\n{mems}')

        component_states = '\n'.join([
                f"{self._agent_name}'s "
                + (construct.name() + ':\n' + construct.state())
                for construct in self._components
        ])

        mood_list = ['happy', 'sad', 'annoyed', 'angry', 'excited', 'neutral', 'bored', 'tired', 'anxious', 'scared']

        prompt.statement(component_states)
        question = (
                f'Given the current situation, how would you describe {self._agent_name}\'s mood? Use the list below to help you: \n {mood_list}'
        )
        if self._clock_now is not None:
                question = f'Current time: {self._clock_now()}.\n{question}'

        current_mood_response = prompt.open_question(
                question,
                answer_prefix=f'{self._agent_name} is currently feeling ',
                max_characters=3000,
                max_tokens=1000,
        )

        print("Current mood response: ", current_mood_response)

        if len(current_mood_response) == 0:
                raise ValueError('No response was given.')

        self._state = current_mood_response

        self._last_chain = prompt
        if self._verbose:
            print(termcolor.colored(self._last_chain.view().text(), 'red'), end='')