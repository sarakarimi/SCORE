from collections import deque
from typing import (
    Any,
    Deque,
    Dict,
    Optional,
    Sequence,
    SupportsFloat,
    Tuple,
    TypeVar,
    Union,
)

import gym
import gymnasium
import numpy as np
import numpy.typing as npt
try:
    import cv2  # this is used in AtariPreprocessing
except ImportError:
    cv2 = None
from gymnasium.spaces import Box as GymnasiumBox
from gymnasium.spaces import Dict as GymnasiumDictSpace
from gymnasium.spaces import Tuple as GymnasiumTuple


NDArray = npt.NDArray[Any]
_ObsType = TypeVar("_ObsType")
_ActType = TypeVar("_ActType")


def _get_keys_from_observation_space(
    observation_space: GymnasiumDictSpace,
) -> Sequence[str]:
    return sorted(list(observation_space.keys()))


def _flat_dict_observation(observation: Dict[str, NDArray]) -> NDArray:
    sorted_keys = sorted(list(observation.keys()))
    return np.concatenate([observation[key] for key in sorted_keys])


class GoalConcatWrapper(
    gymnasium.Wrapper[
        Union[NDArray, Tuple[NDArray, NDArray]],
        _ActType,
        Dict[str, NDArray],
        _ActType,
    ]
):
    r"""GaolConcatWrapper class for goal-conditioned environments.

    This class concatenates a main observation and a goal observation to make a
    single numpy observation output. This is especially useful with environments
    such as AntMaze int the non-hindsight training case.

    Args:
        env (Union[gym.Env, gymnasium.Env]): Goal-conditioned environment.
        observation_key (str): String key of the main observation.
        goal_key (str): String key of the goal observation.
        tuple_observation (bool): Flag to include goals as tuple element.
    """

    _observation_space: Union[GymnasiumBox, GymnasiumTuple]
    _observation_key: str
    _goal_key: str
    _tuple_observation: bool

    def __init__(
        self,
        env: gymnasium.Env[Dict[str, NDArray], _ActType],
        observation_key: str = "observation",
        goal_key: str = "achieved_goal",
        tuple_observation: bool = False,
    ):
        super().__init__(env)
        assert isinstance(env.observation_space, GymnasiumDictSpace)
        self._observation_key = observation_key
        self._goal_key = goal_key
        self._tuple_observation = tuple_observation
        observation_space = env.observation_space[observation_key]
        assert isinstance(observation_space, GymnasiumBox)
        goal_space = env.observation_space[goal_key]
        if isinstance(goal_space, GymnasiumBox):
            goal_space_low = goal_space.low
            goal_space_high = goal_space.high
        elif isinstance(goal_space, GymnasiumDictSpace):
            goal_keys = _get_keys_from_observation_space(goal_space)
            goal_spaces = [goal_space[key] for key in goal_keys]
            goal_space_low = np.concatenate(
                [
                    (
                        [space.low] * space.shape[0]  # type: ignore
                        if np.isscalar(space.low)  # type: ignore
                        else space.low  # type: ignore
                    )
                    for space in goal_spaces
                ]
            )
            goal_space_high = np.concatenate(
                [
                    (
                        [space.high] * space.shape[0]  # type: ignore
                        if np.isscalar(space.high)  # type: ignore
                        else space.high  # type: ignore
                    )
                    for space in goal_spaces
                ]
            )
        else:
            raise ValueError(f"unsupported goal space: {type(goal_space)}")
        if tuple_observation:
            self._observation_space = GymnasiumTuple(
                [observation_space, goal_space]
            )
        else:
            low = np.concatenate([observation_space.low, goal_space_low])
            high = np.concatenate([observation_space.high, goal_space_high])
            self._observation_space = GymnasiumBox(
                low=low,
                high=high,
                shape=low.shape,
                dtype=observation_space.dtype,  # type: ignore
            )

    def step(self, action: _ActType) -> Tuple[
        Union[NDArray, Tuple[NDArray, NDArray]],
        SupportsFloat,
        bool,
        bool,
        Dict[str, Any],
    ]:
        obs, rew, terminal, truncate, info = self.env.step(action)
        goal_obs = obs[self._goal_key]
        if isinstance(goal_obs, dict):
            goal_obs = _flat_dict_observation(goal_obs)
        concat_obs: Union[NDArray, Tuple[NDArray, NDArray]]
        if self._tuple_observation:
            concat_obs = (goal_obs, obs[self._observation_key])
        else:
            concat_obs = np.concatenate([goal_obs, obs[self._observation_key]])
        return concat_obs, rew, terminal, truncate, info

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Union[NDArray, Tuple[NDArray, NDArray]], Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        goal_obs = obs[self._goal_key]
        if isinstance(goal_obs, dict):
            goal_obs = _flat_dict_observation(goal_obs)
        concat_obs: Union[NDArray, Tuple[NDArray, NDArray]]
        if self._tuple_observation:
            concat_obs = (goal_obs, obs[self._observation_key])
        else:
            concat_obs = np.concatenate([goal_obs, obs[self._observation_key]])
        return concat_obs, info