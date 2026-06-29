from typing import Dict
import time

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int, chunk_sleep: float = 0.0):
        self._policy = policy
        self._action_horizon = action_horizon
        self._chunk_sleep = chunk_sleep  # 执行完一个 chunk 后的等待时间
        self._cur_step: int = 0
        self._chunk_id: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        if self._last_results is None:
            if self._chunk_sleep > 0:
                print(f"[ActionChunkBroker] Chunk finished, sleeping {self._chunk_sleep}s...")
                time.sleep(self._chunk_sleep)
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0
            self._chunk_id += 1

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        step = self._cur_step
        results = tree.map_structure(slicer, self._last_results)
        # 与异步 broker 对齐元信息字段，供下游时间轴记录器使用。
        results["_chunk_id"] = self._chunk_id
        results["_chunk_step"] = step
        results["_in_blend"] = False
        if step == 0:
            results["_chunk_actions_full"] = self._last_results.get("actions")
            results["_chunk_velocities_full"] = self._last_results.get("velocities")
            results["_chunk_skip_steps"] = 0
            results["_chunk_blend_window"] = 0
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None
            # 执行完一个完整 chunk 后等待


        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0
        self._chunk_id = 0
