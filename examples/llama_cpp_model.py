from concordia.language_model import language_model
from concordia.utils import measurements as measurements_lib

from langchain.llms import HuggingFacePipeline
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from collections.abc import Collection, Sequence
from string import Template
from typing_extensions import override

import logging
import re
import torch
import llama_cpp


# Set up logging
logging.basicConfig(level=logging.INFO, filename='llama_cpp.log')
logger = logging.getLogger('llama_cpp')

_MAX_MULTIPLE_CHOICE_ATTEMPTS = 3
_MAX_SAMPLE_TEXT_ATTEMPTS = 5

# PROMPT_TEMPLATE = Template("""<s>[INST] <<SYS>>\n$system_message\n<</SYS>>\n$message[/INST]""")
PROMPT_TEMPLATE = Template("$system_message\n\n$message")

# TEXT_SYSTEM_MESSAGE = """Your task is to follow instructions and answer questions correctly and concisely."""
TEXT_SYSTEM_MESSAGE = ""

MULTIPLE_CHOICE_SYSTEM_MESSAGE = """The following is a multiple choice question.
You must always respond with one of the possible choices.
You may not respond with anything else.
"""
# MULTIPLE_CHOICE_SYSTEM_MESSAGE = ""

class LlamaCppLanguageModel(language_model.LanguageModel):
    """Language Model that uses huggingface LLM models."""

    def __init__(
        self,
        model_path: str = "",
        repo_id: str = "",
        filename: str = "",
        measurements: measurements_lib.Measurements | None = None,
        channel: str = language_model.DEFAULT_STATS_CHANNEL,
        logits: bool = False,
        verbose: bool = False,
    ):
        """Initializes the instance.
        
        Args:
          model_name: The language model to use.
          measurements: The measurements object to log usage statistics to.
          channel: The channel to write the statistics to.
          logits: Whether to return logits instead of text. 
            - Returned logits is a tuple of (batch_size, vocab_size) tensors for each output token.
        """
        self._model_name = model_path
        self._measurements = measurements
        self._channel = channel

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert model_path or (repo_id and filename), "Either model_path or repo_id + filename must be provided."
        
        if model_path:
            self._client = llama_cpp.Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                logits_all=True,
                verbose=False,
                chat_format="llama-2",
            )
        else:
            self._client = llama_cpp.Llama.from_pretrained(
                repo_id=repo_id,
                filename=filename,
                n_gpu_layers=-1,
                logits_all=True,
                verbose=False,
                chat_format="llama-2",
            )

        self._logits = logits
        self._verbose = verbose
        self._client

    @override   
    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = language_model.DEFAULT_MAX_TOKENS,
        max_characters: int = language_model.DEFAULT_MAX_CHARACTERS,
        terminators: Collection[str] = language_model.DEFAULT_TERMINATORS,
        temperature: float = language_model.DEFAULT_TEMPERATURE,
        timeout: float = language_model.DEFAULT_TIMEOUT_SECONDS,
        seed: int | None = None,
        system_message = TEXT_SYSTEM_MESSAGE,
    ) -> str:
        
        message = prompt
        # prompt = PROMPT_TEMPLATE.substitute(system_message=system_message, message=message)

        for retry in range(_MAX_SAMPLE_TEXT_ATTEMPTS): 
            try:
                # print(f"Sending prompt to LLM: {message}")
                outputs = self._client.create_chat_completion(
                    messages = [
                        {"role": "user", "content": message}
                    ]
                )
                
                if self._logits:
                    if self._verbose:
                        print("The LLM response is: ")
                        print(outputs["choices"][0]["message"]["content"])
                    # last possible index is 70
                    response = [(self._client._model.token_get_text(i), llama_cpp.llama_get_logits_ith(self._client.ctx, 1)[i]) for i in range(self._client.n_vocab())]
                    
                    # Sort response by logit value
                    response.sort(key=lambda x: x[1], reverse=True)

                    for i in range(5):
                        print(f"{response[i][0]}: {response[i][1]}")

                    # print("The response is: ")
                    # print(response)
                    
                    # response = [(self._client._model.token_get_text(i), llama_cpp.llama_get_logits(self._client.ctx)[i]) for i in range(self._client.n_vocab())]    
                else:
                    response = outputs

            except ValueError as e:
                logger.error(f"Error while calling LLM with input: {prompt}. Attempt {retry+1} failed.")
                if retry == _MAX_SAMPLE_TEXT_ATTEMPTS - 1: 
                    logger.error(f"Max retries exceeded. Raising exception.")
                    raise language_model.InvalidResponseError(prompt)
            else:
                logger.debug(f"Succeeded after {retry+1} attempts.")
                logger.debug(f"Response from LLM: {response}")
                break

        if self._measurements is not None:
            self._measurements.publish_datum(
                self._channel,
                {'raw_text_length': len(response)},
            )
        return response

    @staticmethod
    def extract_choices(text):
        match = re.search(r'\(?(\w)\)', text)
        if match:
            return match.group(1)
        else:
            return None

    @override
    def sample_choice(
        self,
        prompt: str,
        responses: Sequence[str],
        *,
        seed: int | None = None,
    ) -> tuple[int, str, dict[str, float]]:
        max_characters = len(max(responses, key=len))

        attempts = 1
        for _ in range(_MAX_MULTIPLE_CHOICE_ATTEMPTS):
            sample = self.sample_text(
                prompt,
                max_characters=max_characters,
                temperature=0.0,
                seed=seed,
                system_message=MULTIPLE_CHOICE_SYSTEM_MESSAGE,
            )
            answer = self.extract_choices(sample)
            try:
                idx = responses.index(answer)
            except ValueError:
                attempts += 1
                continue
            else:
                if self._measurements is not None:
                    self._measurements.publish_datum(
                        self._channel, {'choices_calls': attempts}
                    )
                debug = {}
                return idx, responses[idx], debug

        logger.error(f"Multiple choice failed after {_MAX_MULTIPLE_CHOICE_ATTEMPTS} attempts.\nLLM Input: {prompt}\nLLM Output: {sample}\nExtracted Answer: {answer}")

        raise language_model.InvalidResponseError(
             f'Too many multiple choice attempts.\nLLM Input: {prompt}\nLLM Output: {sample}'
        )