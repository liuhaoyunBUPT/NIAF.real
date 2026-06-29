"""
异步 Action Chunk Broker

实现推理与执行的解耦：在执行当前 chunk 的过程中，提前触发下一次推理，
避免 chunk 之间的停顿。

核心流程 (以 action_horizon=50, trigger_step=25 为例):
    chunk1:  [0 .... 24]  [25 ..... 39]  [40 .......... 49]
                           ↑ 触发推理     ↑ 推理完成，切换到chunk2
    chunk2:         skip[0...14]  [15..19 blend]  [20 ......... 49]
                                  ↑ blend窗口抵消位置偏移

切换时的Blend修正:
    - skip_steps = k_exec_at_switch - k_exec_at_submit  (推理期间真实执行的步数)
    - gap = q_actual[k_exec] - new_chunk[skip_steps] (当前真实状态 vs 新chunk对应步的预测值)
    - 对 new_chunk 的 [skip_steps, skip_steps + blend_window) 施加 smoothstep 修正
    - blend 只修正关节维度 (index 0-5, 7-12)，不修正夹爪 (index 6, 13)
"""

import copy
import logging
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy


class AsyncActionChunkBroker(_base_policy.BasePolicy):
    """异步 Action Chunk Broker — 推理与执行解耦，支持平滑过渡。

    与同步版 ActionChunkBroker 的接口保持一致 (infer/reset)，可直接替换使用。
    """

    # 14维动作中的关节索引 (不含夹爪 index 6, 13)
    JOINT_INDICES = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int = 50,
        trigger_step: int = 25,
        blend_window: int = 5,
        joint_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            policy: 底层推理策略 (如 WebsocketClientPolicy)，
                    其 infer() 返回 {"actions": (H, 14), "velocities": (H, 14), ...}
            action_horizon: 每个 chunk 的动作步数 H
            trigger_step: 执行到第几步时触发后台推理 (默认 H/2)
            blend_window: 切换时的过渡窗口步数 (smoothstep 修正)
            joint_indices: 需要 blend 修正的关节维度索引，默认 [0..5, 7..12]
        """
        self._policy = policy
        self._action_horizon = action_horizon
        self._trigger_step = trigger_step
        self._blend_window = blend_window
        self._joint_indices = joint_indices if joint_indices is not None else self.JOINT_INDICES

        # ---- 当前 chunk 状态 ----
        self._current_chunk: Optional[Dict[str, np.ndarray]] = None  # 完整推理结果 (H, D)
        self._k_exec: int = 0  # 当前执行到第几步
        self._current_chunk_id: int = 0
        self._blend_start_step: Optional[int] = None
        self._blend_end_step: Optional[int] = None
        self._chunk_meta_emit_step: Optional[int] = None

        # ---- 异步推理 ----
        self._infer_triggered: bool = False  # 本 chunk 是否已触发推理
        self._pending_chunk: Optional[Dict[str, np.ndarray]] = None  # 后台推理完成的新 chunk
        self._infer_submit_k_exec: Optional[int] = None  # 触发后台推理时的执行步索引
        self._lock = threading.Lock()
        self._infer_thread: Optional[threading.Thread] = None

        # ---- 统计 ----
        self._chunk_count: int = 0
        self._switch_count: int = 0

    @override
    def infer(self, obs: Dict) -> Dict:
        """每个控制周期调用一次，返回单步动作 dict。

        Args:
            obs: 当前观测，必须包含 "state" (14维关节状态)

        Returns:
            单步动作 dict: {"actions": (14,), "velocities": (14,), ...}
        """
        # =========== 1. 首次调用：阻塞推理初始化第一个 chunk ===========
        if self._current_chunk is None:
            logging.info("[AsyncBroker] 首次推理 (阻塞)...")
            raw = self._policy.infer(obs)
            self._current_chunk = raw
            self._k_exec = 0
            self._infer_triggered = False
            self._chunk_count += 1
            self._current_chunk_id = self._chunk_count
            self._blend_start_step = None
            self._blend_end_step = None
            self._chunk_meta_emit_step = 0
            logging.info(f"[AsyncBroker] 首个chunk就绪, actions shape: "
                         f"{raw['actions'].shape if 'actions' in raw else 'N/A'}")

        # =========== 2. 在 trigger_step 触发后台推理 ===========
        if self._k_exec == self._trigger_step and not self._infer_triggered:
            self._submit_inference(obs)

        # =========== 3. 检查是否有新 chunk 可以切换 ===========
        switched = False
        with self._lock:
            if self._pending_chunk is not None and self._k_exec >= self._trigger_step:
                switched = self._switch_to_pending(obs)

        # =========== 4. 处理 chunk 耗尽但无新 chunk 的情况 (hold) ===========
        if self._k_exec >= self._action_horizon:
            if not switched:
                # 检查是否有 pending（推理线程可能刚好完成）
                with self._lock:
                    if self._pending_chunk is not None:
                        switched = self._switch_to_pending(obs)

                if not switched:
                    # 保持最后姿态等待
                    logging.warning(f"[AsyncBroker] chunk耗尽，保持最后姿态等待新chunk "
                                    f"(chunk={self._chunk_count})")
                    return self._slice_step(self._action_horizon - 1)

        # =========== 5. 返回当前步动作 ===========
        result = self._slice_step(self._k_exec)
        self._k_exec += 1
        return result

    def _submit_inference(self, obs: Dict) -> None:
        """提交后台推理任务。快照当前观测发送给推理线程。"""
        self._infer_triggered = True
        self._infer_submit_k_exec = self._k_exec
        # 深拷贝 obs 防止主线程后续修改
        obs_snapshot = copy.deepcopy(obs)

        def _infer_worker():
            try:
                t0 = time.time()
                result = self._policy.infer(obs_snapshot)
                dt = time.time() - t0
                with self._lock:
                    self._pending_chunk = result
                logging.info(f"[AsyncBroker] 后台推理完成, 耗时 {dt:.3f}s "
                             f"(约 {int(dt * 30):.0f} steps @30Hz)")
            except Exception as e:
                logging.error(f"[AsyncBroker] 后台推理失败: {e}")

        self._infer_thread = threading.Thread(target=_infer_worker, daemon=True)
        self._infer_thread.start()

    def _switch_to_pending(self, obs: Dict) -> bool:
        """将 pending chunk 切换为当前 chunk，并施加 blend 修正。

        必须在持有 self._lock 的情况下调用。

        Returns:
            True 如果切换成功
        """
        new_chunk = self._pending_chunk
        self._pending_chunk = None

        # 计算跳过的步数：后台推理期间主循环真实执行的步数
        submit_k_exec = self._infer_submit_k_exec
        if submit_k_exec is None:
            submit_k_exec = self._trigger_step
        skip_steps = self._k_exec - submit_k_exec
        if skip_steps < 0:
            skip_steps = 0

        # 如果 skip_steps 超出新 chunk 范围，截断到最后一步
        H = new_chunk["actions"].shape[0]
        if skip_steps >= H:
            logging.warning(f"[AsyncBroker] skip_steps={skip_steps} >= H={H}, 截断到 H-1")
            skip_steps = H - 1

        # 获取当前真实关节状态 (适配 obs["qpos"] 或 obs["state"])
        current_state = np.array(
            obs.get("qpos", obs.get("state", np.zeros(14))), 
            dtype=np.float32
        )

        # 计算 gap: 当前真实状态 vs 新chunk在 skip_steps 处的预测值
        new_actions = new_chunk["actions"].copy()  # (H, 14)
        gap = np.zeros_like(new_actions[0])
        gap[self._joint_indices] = (
            current_state[self._joint_indices] - new_actions[skip_steps, self._joint_indices]
        )

        logging.info(f"[AsyncBroker] 切换chunk: k_exec={self._k_exec}, skip={skip_steps}, "
                      f"gap_norm={np.linalg.norm(gap[self._joint_indices]):.5f}")

        # 施加 blend 修正：在 [skip_steps, skip_steps + blend_window) 窗口内
        # 用 smoothstep 将 gap 从 1.0 衰减到 0.0
        blend_end = min(skip_steps + self._blend_window, H)
        for k in range(skip_steps, blend_end):
            m = k - skip_steps  # 窗口内的索引 [0, blend_window)
            alpha = 1.0 - self._smoothstep(m, self._blend_window)
            new_actions[k, self._joint_indices] += gap[self._joint_indices] * alpha

        # 更新 chunk（深拷贝后修改 actions）
        corrected_chunk = {}
        for key, val in new_chunk.items():
            if key == "actions":
                corrected_chunk[key] = new_actions
            else:
                corrected_chunk[key] = val  # velocities 等字段不修正

        self._current_chunk = corrected_chunk
        self._k_exec = skip_steps
        self._infer_triggered = False
        self._infer_submit_k_exec = None
        self._chunk_count += 1
        self._current_chunk_id = self._chunk_count
        self._blend_start_step = skip_steps
        self._blend_end_step = blend_end
        self._chunk_meta_emit_step = skip_steps
        self._switch_count += 1

        return True

    def _slice_step(self, step: int) -> Dict:
        """从当前 chunk 中切出第 step 步的单步动作。"""
        def slicer(x):
            if isinstance(x, np.ndarray) and x.ndim >= 1:
                idx = min(step, x.shape[0] - 1)
                return x[idx, ...]
            return x

        result = {}
        for key, val in self._current_chunk.items():
            result[key] = slicer(val)
        in_blend = (
            self._blend_start_step is not None
            and self._blend_end_step is not None
            and self._blend_start_step <= step < self._blend_end_step
        )
        result["_chunk_id"] = self._current_chunk_id
        result["_chunk_step"] = step
        result["_in_blend"] = in_blend
        if self._chunk_meta_emit_step is not None and step == self._chunk_meta_emit_step:
            result["_chunk_actions_full"] = self._current_chunk.get("actions")
            result["_chunk_velocities_full"] = self._current_chunk.get("velocities")
            result["_chunk_skip_steps"] = int(self._blend_start_step or 0)
            if self._blend_start_step is not None and self._blend_end_step is not None:
                result["_chunk_blend_window"] = int(self._blend_end_step - self._blend_start_step)
            else:
                result["_chunk_blend_window"] = 0
            self._chunk_meta_emit_step = None
        return result

    @staticmethod
    def _smoothstep(m: int, M: int) -> float:
        """Smoothstep 函数: t ∈ [0, 1] → 3t² - 2t³

        当 m=0 返回 0.0, m=M 返回 1.0。
        用于 blend 权重: alpha = 1 - smoothstep(m, M)
            → m=0 时 alpha=1.0 (完全用 gap 修正)
            → m=M 时 alpha=0.0 (不修正)
        """
        if M <= 0:
            return 1.0
        t = max(0.0, min(1.0, m / M))
        return t * t * (3.0 - 2.0 * t)

    @override
    def reset(self) -> None:
        """重置 broker 状态。等待后台推理线程结束。"""
        # 等待推理线程结束
        if self._infer_thread is not None and self._infer_thread.is_alive():
            logging.info("[AsyncBroker] 等待推理线程结束...")
            self._infer_thread.join(timeout=10.0)

        self._policy.reset()
        self._current_chunk = None
        self._k_exec = 0
        self._current_chunk_id = 0
        self._blend_start_step = None
        self._blend_end_step = None
        self._chunk_meta_emit_step = None
        self._infer_triggered = False
        self._infer_submit_k_exec = None
        with self._lock:
            self._pending_chunk = None
        self._infer_thread = None

        logging.info(f"[AsyncBroker] 已重置 (共执行 {self._chunk_count} 个chunk, "
                     f"异步切换 {self._switch_count} 次)")
        self._chunk_count = 0
        self._switch_count = 0
