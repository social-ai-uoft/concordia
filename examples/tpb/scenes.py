import datetime
from typing import Mapping, Sequence, Callable, Any

from concordia.typing import scene as scene_lib

from examples.tpb import agent as tpb_agent

class Scene:
  """
  A scene that takes in a list of premises as context for the agent,
  and returns a list of scenes
  """

  @classmethod
  def build_scenes(
    cls,
    scene_dict: Mapping[
      str, Mapping[
        str, Sequence[str | Callable[[str], str]]
      ]
    ],
    scene_order: Sequence[
      Mapping[str, tuple[datetime.datetime, Sequence[tpb_agent.AgentConfig], int]]
    ]
  ) -> list[scene_lib.SceneSpec]:
    """
    Build a list of scenes from a dictionary of premises.

    Args:
      scene_dict: A mapping between scene types and scene premises.
      scene_order: A list of scenes in order, each with a dictionary specifying the time, configs, and num_rounds.

    Returns:
      A list of scenes.
    """

    scene_types = {}
    for scene_type, spec in scene_dict.items():
      scene_types[scene_type] = scene_lib.SceneTypeSpec(
        name=scene_type,
        premise=spec
      )
    scenes = []
    for scene in scene_order:
      scene = scene_lib.SceneSpec(
        scene_type=scene_types[scene['scene_type']],
        start_time=scene['start_time'],
        participant_configs=scene['participant_configs'],
        num_rounds=scene['num_rounds']
      )
      scenes.append(scene)

    return scenes
