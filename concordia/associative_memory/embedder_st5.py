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


"""English sentence embedding class using ST5.

Ni, J., Ábrego, G.H., Constant, N., Ma, J., Hall, K.B., Cer, D. and Yang, Y.,
2021. Sentence-t5: Scalable sentence encoders from pre-trained text-to-text
models. arXiv preprint arXiv:2108.08877.
"""

from collections.abc import Callable

import numpy as np

import tensorflow as tf
import tensorflow_hub as hub

# import tensorflow_text

ST5 = False

if ST5:
    DEFAULT_ENCODER_URL = "https://tfhub.dev/google/sentence-t5/st5-base/1"

    # `import tensorflow_text` required for embedder to work
    # See https://github.com/tensorflow/tensorflow/issues/38597
    # del tensorflow_text

    class EmbedderST5(Callable):
        """Embeds text using ST5."""

        def __init__(self, hub_url=DEFAULT_ENCODER_URL):
            self._encoder = hub.KerasLayer(hub_url)

        def __call__(self, text: str) -> np.ndarray:
            english_sentences = tf.constant([text])
            (batched_embedding,) = self._encoder(english_sentences)
            return np.squeeze(batched_embedding.numpy())

else:
    # URL for Universal Sentence Encoder on TensorFlow Hub
    DEFAULT_ENCODER_URL = "https://tfhub.dev/google/universal-sentence-encoder/4"

    class EmbedderST5(Callable):
        """Embeds text using Universal Sentence Encoder."""

        def __init__(self, hub_url=DEFAULT_ENCODER_URL):
            # Load the model as a Keras Layer
            self._encoder = hub.KerasLayer(hub_url)

        def __call__(self, text: str) -> np.ndarray:
            # Process the input text
            english_sentences = tf.constant([text])
            # Generate embeddings for the input
            embeddings = self._encoder(english_sentences)
            # Convert the embeddings to a NumPy array and remove extra dimensions
            return np.squeeze(embeddings.numpy())
