#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filepath: /home/agilex/code/openpi-agilex/examples/agilex_cobot_magic/monitor/monitor_joints_left_mit.py
"""
MIT模式下的左臂关节状态监控工具 (无需ROS)

直接从 Piper SDK 获取关节状态并保存为CSV

支持两种录制模式：
1. 普通模式：只录制实际关节位置（puppet）
2. 对比模式：同时录制目标指令（master）和实际位置（puppet）

单位说明 (来自 piper_sdk):
- GetArmJointMsgs().joint_state.joint_x: 0.001 度
- GetArmHighSpdInfoMsgs().motor_x.motor_speed: 0.001 rad/s
- GetArmHighSpdInfoMsgs().motor_x.effort: 0.001 N·m
- GetArmGripperMsgs().gripper_state.grippers_angle: 0.001 mm
"""

import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import sys
import os
import csv
import argparse
import time
import math
import threading
from typing import Optional, Tuple, List

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    try:
        from piper_sdk import C_PiperInterface as C_PiperInterface_V2
    except ImportError as e:
        print("错误: 无法导入 piper_sdk")
        print(f"详细信息: {e}")
        sys.exit(1)

# 单位转换常量
DEG_TO_RAD = math.pi / 180.0  # 度 -> rad
MDEG_TO_RAD = DEG_TO_RAD / 1000.0  # 0.001度 -> rad (= 0.00001745329...)

# 全局数据存储
data_records = []        # puppet 实际位置/速度
master_data_records = [] # master 目标指令（位置+速度）
start_time = None
recording = False
lock = threading.Lock()
silent_mode = False      # 静默模式，不打印保存信息


class SDKJointMonitor:
    """直接从Piper SDK读取左臂关节状态"""
    
    def __init__(self, can_port: str = "can0"):
        self.can_port = can_port
        self.piper: Optional[C_PiperInterface_V2] = None
        self.connected = False
        
    def connect(self) -> bool:
        """连接到机械臂"""
        try:
            if not silent_mode:
                print(f"正在连接左臂 ({self.can_port})...")
            self.piper = C_PiperInterface_V2(self.can_port)
            self.piper.ConnectPort()
            self.connected = True
            if not silent_mode:
                print(f"左臂连接成功")
            return True
        except Exception as e:
            if not silent_mode:
                print(f"连接失败: {e}")
            return False
    
    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取关节状态
        
        Returns:
            (positions, velocities): 
                positions - 7个值 [joint1-6 (rad), gripper (m)]
                velocities - 7个值 [joint1-6 (rad/s), gripper_vel (0)]
        """
        if not self.connected or self.piper is None:
            return np.zeros(7), np.zeros(7)
        
        try:
            # 获取关节角度 (0.001度 -> rad)
            joint_msgs = self.piper.GetArmJointMsgs()
            positions = np.array([
                joint_msgs.joint_state.joint_1 * MDEG_TO_RAD,
                joint_msgs.joint_state.joint_2 * MDEG_TO_RAD,
                joint_msgs.joint_state.joint_3 * MDEG_TO_RAD,
                joint_msgs.joint_state.joint_4 * MDEG_TO_RAD,
                joint_msgs.joint_state.joint_5 * MDEG_TO_RAD,
                joint_msgs.joint_state.joint_6 * MDEG_TO_RAD,
            ])
            
            # 获取夹爪位置 (0.001mm -> m)
            gripper_msgs = self.piper.GetArmGripperMsgs()
            gripper_pos = gripper_msgs.gripper_state.grippers_angle / 1_000_000.0
            positions = np.append(positions, gripper_pos)
            
            # 获取速度 (0.001 rad/s -> rad/s)
            high_spd_msgs = self.piper.GetArmHighSpdInfoMsgs()
            velocities = np.array([
                high_spd_msgs.motor_1.motor_speed / 1000.0,
                high_spd_msgs.motor_2.motor_speed / 1000.0,
                high_spd_msgs.motor_3.motor_speed / 1000.0,
                high_spd_msgs.motor_4.motor_speed / 1000.0,
                high_spd_msgs.motor_5.motor_speed / 1000.0,
                high_spd_msgs.motor_6.motor_speed / 1000.0,
                0.0  # 夹爪速度 (SDK不提供)
            ])
            
            return positions, velocities
            
        except Exception as e:
            if not silent_mode:
                print(f"获取状态失败: {e}")
            return np.zeros(7), np.zeros(7)
    
    def disconnect(self):
        """断开连接"""
        if self.piper is not None:
            try:
                self.piper.DisconnectPort()
                if not silent_mode:
                    print(f"左臂已断开连接")
            except Exception as e:
                if not silent_mode:
                    print(f"断开连接失败: {e}")
        self.connected = False


# ============ 全局录制接口（供推理代码调用） ============

_global_monitor: Optional[SDKJointMonitor] = None
_global_start_time: Optional[float] = None
_global_recording = False
_global_timeline_recorder = None


class TimelineRecorder:
    """内存时间轴记录器：结束时直接输出连续 chunk 可视化 PNG。"""

    def __init__(self, output_file: str):
        self.output_file = output_file
        self.exec_records = []
        self.model_chunks = {}

    def record_model_chunk(
        self,
        rel_time: float,
        dt: float,
        chunk_id: int,
        chunk_step: int,
        chunk_actions_full: Optional[np.ndarray],
        chunk_velocities_full: Optional[np.ndarray],
        chunk_skip_steps: int,
        chunk_blend_window: int,
    ) -> None:
        if chunk_id < 0 or chunk_actions_full is None:
            return
        if chunk_id in self.model_chunks:
            return

        actions = np.asarray(chunk_actions_full, dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < 7:
            return

        if chunk_velocities_full is not None:
            velocities = np.asarray(chunk_velocities_full, dtype=np.float32)
            if velocities.ndim != 2 or velocities.shape[1] < 7:
                velocities = np.zeros_like(actions)
        else:
            velocities = np.zeros_like(actions)

        h = int(actions.shape[0])
        step_index = max(0, min(int(chunk_step), h - 1))
        t_chunk0 = float(rel_time - step_index * dt)

        self.model_chunks[chunk_id] = {
            "chunk_id": int(chunk_id),
            "t0": t_chunk0,
            "dt": float(dt),
            "actions": actions[:, :7].copy(),
            "velocities": velocities[:, :7].copy(),
            "skip_steps": int(max(0, chunk_skip_steps)),
            "blend_window": int(max(0, chunk_blend_window)),
        }

    def record_exec_step(
        self,
        rel_time: float,
        qpos7: np.ndarray,
        qvel7: Optional[np.ndarray],
        chunk_id: int,
        in_overlap: bool,
    ) -> None:
        vel = np.zeros(7, dtype=np.float32) if qvel7 is None else np.asarray(qvel7, dtype=np.float32)
        self.exec_records.append(
            {
                "time": float(rel_time),
                "pos": np.asarray(qpos7, dtype=np.float32).tolist(),
                "vel": vel.tolist(),
                "chunk_id": int(chunk_id),
                "in_overlap": bool(in_overlap),
            }
        )

    def save_plots(self) -> None:
        if not self.model_chunks and not self.exec_records:
            return

        root, _ = os.path.splitext(self.output_file)
        model_png = f"{root}_timeline_model.png"
        exec_png = f"{root}_timeline_exec.png"

        chunk_regions, overlap_regions = self._build_model_regions()

        if self.model_chunks:
            self._plot_model_chunk_timeline("Model Output Timeline", model_png, chunk_regions, overlap_regions)
        if self.exec_records:
            self._plot_exec_timeline("Execution Timeline", exec_png, chunk_regions, overlap_regions)

        if self.model_chunks:
            print(f"[TimelineRecorder] 已保存: {model_png}")
        if self.exec_records:
            print(f"[TimelineRecorder] 已保存: {exec_png}")

    def _write_csv_rows(self, path: str, header: list[str], rows: list[list]) -> None:
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)

    def save_raw_csvs(self) -> None:
        """导出时间轴原始数据到5个CSV，供离线分析与复现。"""
        if not self.model_chunks and not self.exec_records:
            return

        root, _ = os.path.splitext(self.output_file)
        joint_header = [f"joint{i}" for i in range(7)]

        # 1) 执行侧位置与速度（左臂7维）
        exec_sorted = sorted(self.exec_records, key=lambda x: float(x["time"]))
        exec_pos_rows = [[float(r["time"])] + list(np.asarray(r["pos"], dtype=np.float32)[:7]) for r in exec_sorted]
        exec_vel_rows = [[float(r["time"])] + list(np.asarray(r["vel"], dtype=np.float32)[:7]) for r in exec_sorted]

        self._write_csv_rows(
            f"{root}_exec_pos.csv",
            ["time"] + joint_header,
            exec_pos_rows,
        )
        self._write_csv_rows(
            f"{root}_exec_vel.csv",
            ["time"] + joint_header,
            exec_vel_rows,
        )

        # 2) 模型侧chunk展开后的位置与速度（左臂7维）
        model_pos_rows = []
        model_vel_rows = []
        meta_rows = []
        sorted_chunks = sorted(self.model_chunks.values(), key=lambda x: float(x["t0"]))
        for chunk in sorted_chunks:
            chunk_id = int(chunk["chunk_id"])
            t0 = float(chunk["t0"])
            dt = float(chunk["dt"])
            actions = np.asarray(chunk["actions"], dtype=np.float32)
            velocities = np.asarray(chunk["velocities"], dtype=np.float32)
            h = int(actions.shape[0])

            meta_rows.append(
                [
                    chunk_id,
                    t0,
                    dt,
                    h,
                    int(chunk["skip_steps"]),
                    int(chunk["blend_window"]),
                ]
            )

            for step in range(h):
                t = t0 + step * dt
                model_pos_rows.append([t, chunk_id, step] + list(actions[step, :7]))
                model_vel_rows.append([t, chunk_id, step] + list(velocities[step, :7]))

        self._write_csv_rows(
            f"{root}_model_pos.csv",
            ["time", "chunk_id", "step_in_chunk"] + joint_header,
            model_pos_rows,
        )
        self._write_csv_rows(
            f"{root}_model_vel.csv",
            ["time", "chunk_id", "step_in_chunk"] + joint_header,
            model_vel_rows,
        )
        self._write_csv_rows(
            f"{root}_chunk_meta.csv",
            ["chunk_id", "t0", "dt", "horizon", "skip_steps", "blend_window"],
            meta_rows,
        )

        print(f"[TimelineRecorder] 已保存: {root}_exec_pos.csv")
        print(f"[TimelineRecorder] 已保存: {root}_exec_vel.csv")
        print(f"[TimelineRecorder] 已保存: {root}_model_pos.csv")
        print(f"[TimelineRecorder] 已保存: {root}_model_vel.csv")
        print(f"[TimelineRecorder] 已保存: {root}_chunk_meta.csv")

    def _build_model_regions(self):
        chunks = sorted(self.model_chunks.values(), key=lambda x: x["t0"])
        chunk_regions = []
        overlap_regions = []
        for idx, chunk in enumerate(chunks):
            h = int(chunk["actions"].shape[0])
            dt = float(chunk["dt"])
            t0 = float(chunk["t0"])
            t1 = float(t0 + max(h - 1, 0) * dt)
            chunk_regions.append((t0, t1, idx))

            skip = int(chunk["skip_steps"])
            blend = int(chunk["blend_window"])
            if blend > 0:
                overlap_start = t0 + max(0, min(skip, h)) * dt
                overlap_end = t0 + max(0, min(skip + blend, h)) * dt
                if overlap_end > overlap_start:
                    overlap_regions.append((overlap_start, overlap_end, idx))
        return chunk_regions, overlap_regions

    def _plot_model_chunk_timeline(self, title: str, out_path: str, chunk_regions: list, overlap_regions: list) -> None:
        fig, axes = plt.subplots(7, 2, figsize=(16, 18), sharex=True)
        chunk_colors = ("#4C78A8", "#72B7B2")

        joint_names = [f"joint{i}" for i in range(6)] + ["gripper"]
        sorted_chunks = sorted(self.model_chunks.values(), key=lambda x: x["t0"])

        for row in range(7):
            ax_pos = axes[row, 0]
            ax_vel = axes[row, 1]

            for t0, t1, region_idx in chunk_regions:
                color = chunk_colors[region_idx % 2]
                ax_pos.axvspan(t0, t1, color=color, alpha=0.10, linewidth=0)
                ax_vel.axvspan(t0, t1, color=color, alpha=0.10, linewidth=0)

            for t0, t1, _ in overlap_regions:
                ax_pos.axvspan(t0, t1, color="red", alpha=0.18, linewidth=0)
                ax_vel.axvspan(t0, t1, color="red", alpha=0.18, linewidth=0)

            for idx, chunk in enumerate(sorted_chunks):
                dt = float(chunk["dt"])
                h = int(chunk["actions"].shape[0])
                t = chunk["t0"] + np.arange(h, dtype=np.float32) * dt
                c = chunk_colors[idx % 2]
                ax_pos.plot(t, chunk["actions"][:, row], linewidth=0.9, color=c, alpha=0.95)
                ax_vel.plot(t, chunk["velocities"][:, row], linewidth=0.9, color=c, alpha=0.95)

            ax_pos.set_ylabel(joint_names[row])
            ax_pos.grid(alpha=0.25)
            ax_vel.grid(alpha=0.25)

            if row < 6:
                ax_pos.set_ylim(-float(np.pi), float(np.pi))
            if row == 0:
                ax_pos.set_title("Position")
                ax_vel.set_title("Velocity")

        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    def _plot_exec_timeline(self, title: str, out_path: str, chunk_regions: list, overlap_regions: list) -> None:
        arr_t = np.array([r["time"] for r in self.exec_records], dtype=np.float32)
        arr_pos = np.array([r["pos"] for r in self.exec_records], dtype=np.float32)
        arr_vel = np.array([r["vel"] for r in self.exec_records], dtype=np.float32)

        fig, axes = plt.subplots(7, 2, figsize=(16, 18), sharex=True)
        chunk_colors = ("#4C78A8", "#72B7B2")
        joint_names = [f"joint{i}" for i in range(6)] + ["gripper"]

        for row in range(7):
            ax_pos = axes[row, 0]
            ax_vel = axes[row, 1]

            for t0, t1, region_idx in chunk_regions:
                color = chunk_colors[region_idx % 2]
                ax_pos.axvspan(t0, t1, color=color, alpha=0.10, linewidth=0)
                ax_vel.axvspan(t0, t1, color=color, alpha=0.10, linewidth=0)

            for t0, t1, _ in overlap_regions:
                ax_pos.axvspan(t0, t1, color="red", alpha=0.18, linewidth=0)
                ax_vel.axvspan(t0, t1, color="red", alpha=0.18, linewidth=0)

            ax_pos.plot(arr_t, arr_pos[:, row], linewidth=1.0, color="#1f77b4")
            ax_vel.plot(arr_t, arr_vel[:, row], linewidth=1.0, color="#ff7f0e")

            ax_pos.set_ylabel(joint_names[row])
            ax_pos.grid(alpha=0.25)
            ax_vel.grid(alpha=0.25)

            if row < 6:
                ax_pos.set_ylim(-float(np.pi), float(np.pi))
            if row == 0:
                ax_pos.set_title("Position")
                ax_vel.set_title("Velocity")

        axes[-1, 0].set_xlabel("Time (s)")
        axes[-1, 1].set_xlabel("Time (s)")
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)


def init_monitor(can_port: str = "can_left", silent: bool = True, start_time: Optional[float] = None) -> bool:
    """
    初始化全局监控器（供推理代码调用）
    
    Args:
        can_port: CAN端口名称
        silent: 是否静默模式（不打印信息）
    
    用法:
        from monitor.monitor_joints_left_mit import init_monitor, record_step, save_recorded_data
        
        init_monitor("can_left", silent=True)
        # 在推理循环中:
        record_step(master_action=action[:7], master_velocity=action_vel[:7], 
                   puppet_qpos=qpos[:7])
        # 结束时:
        save_recorded_data("output.csv")
    """
    global _global_monitor, _global_start_time, _global_recording, _global_timeline_recorder
    global data_records, master_data_records, silent_mode
    
    silent_mode = silent
    data_records = []
    master_data_records = []
    
    _global_monitor = SDKJointMonitor(can_port)
    if not _global_monitor.connect():
        return False
    
    _global_start_time = float(start_time) if start_time is not None else time.time()
    _global_recording = True
    _global_timeline_recorder = None
    return True


def init_timeline_recorder(output_file: str) -> None:
    global _global_timeline_recorder
    _global_timeline_recorder = TimelineRecorder(output_file=output_file)


def record_timeline_step(
    dt: float,
    model_pos7: np.ndarray,
    model_vel7: Optional[np.ndarray],
    exec_qpos7: np.ndarray,
    exec_qvel7: Optional[np.ndarray],
    chunk_id: int,
    in_overlap: bool,
    chunk_step: int = -1,
    chunk_actions_full: Optional[np.ndarray] = None,
    chunk_velocities_full: Optional[np.ndarray] = None,
    chunk_skip_steps: int = 0,
    chunk_blend_window: int = 0,
) -> None:
    global _global_start_time, _global_timeline_recorder
    if _global_timeline_recorder is None or _global_start_time is None:
        return

    rel_time = time.time() - _global_start_time
    with lock:
        _global_timeline_recorder.record_model_chunk(
            rel_time=rel_time,
            dt=dt,
            chunk_id=chunk_id,
            chunk_step=chunk_step,
            chunk_actions_full=chunk_actions_full,
            chunk_velocities_full=chunk_velocities_full,
            chunk_skip_steps=chunk_skip_steps,
            chunk_blend_window=chunk_blend_window,
        )
        _global_timeline_recorder.record_exec_step(
            rel_time=rel_time,
            qpos7=exec_qpos7,
            qvel7=exec_qvel7,
            chunk_id=chunk_id,
            in_overlap=in_overlap,
        )


def save_timeline_plots() -> None:
    global _global_timeline_recorder
    if _global_timeline_recorder is None:
        return
    _global_timeline_recorder.save_plots()
    try:
        _global_timeline_recorder.save_raw_csvs()
    except Exception as e:
        print(f"[TimelineRecorder] 导出CSV失败: {e}")


def shutdown_monitor() -> None:
    global _global_monitor, _global_recording
    _global_recording = False
    if _global_monitor is not None:
        try:
            _global_monitor.disconnect()
        except Exception:
            pass
    _global_monitor = None


def record_step(master_action: Optional[np.ndarray] = None, 
                master_velocity: Optional[np.ndarray] = None,
                puppet_qpos: Optional[np.ndarray] = None):
    """
    记录一步数据（供推理代码调用）
    
    Args:
        master_action: 目标指令位置 (7维: 6关节 + 1夹爪), 如果为 None 则不记录 master
        master_velocity: 目标指令速度 (7维), 如果为 None 则记录 master 但速度为 0
        puppet_qpos: 实际位置 (7维), 如果为 None 则从 SDK 读取
    """
    global _global_start_time, _global_recording, _global_monitor
    
    if not _global_recording or _global_start_time is None:
        return
    
    rel_time = time.time() - _global_start_time
    
    # 获取 puppet 数据
    if puppet_qpos is not None:
        positions = np.array(puppet_qpos)
        velocities = np.zeros(len(puppet_qpos))
        # 如果可以，从 SDK 获取速度
        if _global_monitor is not None and _global_monitor.connected:
            _, velocities = _global_monitor.get_joint_states()
    elif _global_monitor is not None:
        positions, velocities = _global_monitor.get_joint_states()
    else:
        return
    
    # 记录 puppet 数据
    puppet_rec = {
        'time': rel_time,
        'pos': list(positions),
        'vel': list(velocities)
    }
    with lock:
        data_records.append(puppet_rec)
    
    # 记录 master 数据（位置+速度）
    if master_action is not None:
        # 如果没有提供速度，则使用零速度
        if master_velocity is None:
            master_velocity = np.zeros(len(master_action))
        
        master_rec = {
            'time': rel_time,
            'pos': list(master_action),
            'vel': list(master_velocity)
        }
        with lock:
            master_data_records.append(master_rec)


def stop_recording():
    """停止录制"""
    global _global_recording
    _global_recording = False


def save_recorded_data(filename: str):
    """
    保存录制的数据
    
    Args:
        filename: 输出文件名，会生成:
            - {filename}.csv - puppet 位置
            - {filename}_vel.csv - puppet 速度
            - {filename}_acc.csv - puppet 加速度
            - {filename}_master.csv - master 指令位置
            - {filename}_master_vel.csv - master 指令速度
            - {filename}_master_acc.csv - master 指令加速度
    """
    global _global_monitor, _global_recording
    
    _global_recording = False
    
    if _global_monitor is not None:
        _global_monitor.disconnect()
    
    # 保存 puppet 数据
    save_csv(filename)
    
    # 保存 master 数据（包括速度和加速度）
    if master_data_records:
        root, ext = os.path.splitext(filename)
        master_filename = f"{root}_master{ext}"
        save_csv_with_velocity(master_filename, master_data_records)


# ============ 保存函数 ============

def recording_thread(monitor: SDKJointMonitor, sample_rate: float = 50.0):
    """录制线程"""
    global start_time, recording, data_records
    
    dt = 1.0 / sample_rate
    start_time = time.time()
    
    if not silent_mode:
        print(f"开始录制 (采样率: {sample_rate} Hz)")
        print("按 Ctrl+C 停止录制")
    
    sample_count = 0
    while recording:
        loop_start = time.time()
        
        # 读取状态
        positions, velocities = monitor.get_joint_states()
        
        # 计算相对时间
        rel_time = time.time() - start_time
        
        # 保存数据
        rec = {
            'time': rel_time,
            'pos': list(positions),
            'vel': list(velocities)
        }
        
        with lock:
            data_records.append(rec)
        
        sample_count += 1
        if not silent_mode and sample_count % int(sample_rate) == 0:  # 每秒打印一次
            print(f"[{rel_time:.1f}s] pos[0:3]={positions[:3].round(3)}, vel[0:3]={velocities[:3].round(3)}")
        
        # 控制采样率
        elapsed = time.time() - loop_start
        sleep_time = max(0, dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


def calculate_derivative(times, values):
    """计算离散导数 (p(t) - p(t-1)) / dt"""
    derivs = []
    if len(times) < 2 or not values:
        if values:
            return [[0.0] * len(values[0])] * len(values)
        return []
    
    num_joints = len(values[0])
    derivs.append([0.0] * num_joints)  # 初始导数为0
    
    for i in range(1, len(times)):
        dt = times[i] - times[i-1]
        row = []
        for j in range(num_joints):
            if dt > 1e-6:
                val = (values[i][j] - values[i-1][j]) / dt
            else:
                val = 0.0
            row.append(val)
        derivs.append(row)
    
    return derivs


def save_single_csv(filename, times, data, header):
    """保存单个CSV文件"""
    if not silent_mode:
        print(f"保存 {len(data)} 条记录到 {filename}...")
    try:
        # 确保目录存在
        dir_path = os.path.dirname(filename)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t, row in zip(times, data):
                writer.writerow([t] + row)
        if not silent_mode:
            print(f"✓ 成功保存到 {filename}")
    except Exception as e:
        if not silent_mode:
            print(f"✗ 保存CSV失败 {filename}: {e}")


def save_position_only_csv(filename, records):
    """保存仅包含位置的CSV (用于master指令，已废弃，改用save_csv_with_velocity)"""
    if not records:
        if not silent_mode:
            print("没有master数据可保存")
        return
    
    times = [r['time'] for r in records]
    positions = [r['pos'] for r in records]
    
    num_joints = len(positions[0])
    header = ["time"] + [f"joint{i}" for i in range(num_joints)]
    
    save_single_csv(filename, times, positions, header)


def save_csv_with_velocity(filename, records):
    """保存包含位置和速度的CSV (用于master数据，生成3个文件)"""
    if not records:
        if not silent_mode:
            print("没有数据可保存")
        return
    
    times = [r['time'] for r in records]
    positions = [r['pos'] for r in records]
    velocities = [r['vel'] for r in records]
    
    # 确定表头
    num_joints = len(positions[0])
    header = ["time"] + [f"joint{i}" for i in range(num_joints)]
    
    # 1. 保存位置
    save_single_csv(filename, times, positions, header)
    
    root, ext = os.path.splitext(filename)
    
    # 2. 保存速度
    if not silent_mode:
        print(f"速度来源: [Master指令]")
    vel_filename = f"{root}_vel{ext}"
    save_single_csv(vel_filename, times, velocities, header)
    
    # 3. 保存加速度 (速度的导数)
    if not silent_mode:
        print(f"加速度来源: [计算导数]")
    accelerations = calculate_derivative(times, velocities)
    acc_filename = f"{root}_acc{ext}"
    save_single_csv(acc_filename, times, accelerations, header)


def save_csv(filename):
    """保存位置、速度、加速度到三个CSV文件"""
    if not data_records:
        if not silent_mode:
            print("没有数据可保存")
        return
    
    times = [r['time'] for r in data_records]
    positions = [r['pos'] for r in data_records]
    velocities = [r['vel'] for r in data_records]
    
    # 确定表头: 6个关节 + 1个夹爪
    num_joints = len(positions[0])
    header = ["time"] + [f"joint{i}" for i in range(num_joints)]
    
    # 1. 保存位置
    save_single_csv(filename, times, positions, header)
    
    root, ext = os.path.splitext(filename)
    
    # 2. 保存速度 (直接从SDK获取)
    if not silent_mode:
        print(f"速度来源: [SDK实时读取]")
    vel_filename = f"{root}_vel{ext}"
    save_single_csv(vel_filename, times, velocities, header)
    
    # 3. 保存加速度 (速度的导数)
    if not silent_mode:
        print(f"加速度来源: [计算导数]")
    accelerations = calculate_derivative(times, velocities)
    acc_filename = f"{root}_acc{ext}"
    save_single_csv(acc_filename, times, accelerations, header)


def record_mode(filename, can_port, sample_rate):
    """录制模式"""
    global recording, data_records
    
    # 清空之前的数据
    data_records = []
    
    # 初始化监控器
    monitor = SDKJointMonitor(can_port=can_port)
    
    if not monitor.connect():
        if not silent_mode:
            print("连接失败,退出")
        return
    
    # 等待一下让连接稳定
    time.sleep(0.5)
    
    # 启动录制线程
    recording = True
    thread = threading.Thread(target=recording_thread, args=(monitor, sample_rate))
    thread.daemon = True
    thread.start()
    
    try:
        # 等待用户中断
        while recording:
            time.sleep(0.1)
    except KeyboardInterrupt:
        if not silent_mode:
            print("\n\n正在停止录制...")
    finally:
        # 无论如何都要保存数据
        recording = False
        thread.join(timeout=2.0)
        
        # 保存数据
        if data_records:
            save_csv(filename)
            if not silent_mode:
                print(f"\n录制完成! 共录制 {len(data_records)} 条数据, 时长 {data_records[-1]['time']:.2f}s")
        else:
            if not silent_mode:
                print("\n没有数据可保存")
        
        # 断开连接
        monitor.disconnect()


def plot_mode(filename):
    """绘图模式"""
    if not os.path.exists(filename):
        print(f"文件不存在: {filename}")
        return

    print(f"读取 {filename}...")
    try:
        data = []
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                data.append([float(x) for x in row])
        
        data_np = np.array(data)
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    if data_np.size == 0:
        print("数据为空")
        return

    time_axis = data_np[:, 0]
    joints_data = data_np[:, 1:]
    
    num_vals = joints_data.shape[1]
    
    print(f"绘制 {num_vals} 条曲线, 时长 {time_axis[-1]:.2f} 秒...")

    fig, axes = plt.subplots(num_vals, 1, sharex=True, figsize=(12, 2.5 * num_vals))
    if num_vals == 1:
        axes = [axes]
    
    for i in range(num_vals):
        ax = axes[i]
        label_name = f"joint{i}" if i < 6 else "gripper"
        
        ax.plot(time_axis, joints_data[:, i], label=label_name, linewidth=0.8)
        ax.set_ylabel(label_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 前6个关节设置Y轴范围为[-pi, pi] (仅位置数据)
        if i < 6 and not ("_vel" in filename or "_acc" in filename):
            ax.set_ylim(-np.pi, np.pi)
            ax.axhline(y=np.pi, color='r', linestyle='--', alpha=0.3)
            ax.axhline(y=-np.pi, color='r', linestyle='--', alpha=0.3)
            
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.1)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"Left Arm Joint States - {os.path.basename(filename)}")
    plt.tight_layout()
    
    plot_filename = filename.replace('.csv', '.png')
    if plot_filename == filename:
        plot_filename += ".png"
        
    plt.savefig(plot_filename, dpi=150)
    print(f"图表已保存到 {plot_filename}")


def compare_plot_mode(puppet_file, master_file):
    """绘制 master vs puppet 对比图"""
    if not os.path.exists(puppet_file):
        print(f"Puppet文件不存在: {puppet_file}")
        return
    if not os.path.exists(master_file):
        print(f"Master文件不存在: {master_file}")
        return

    print(f"读取 puppet 数据: {puppet_file}")
    print(f"读取 master 数据: {master_file}")
    
    try:
        # 读取 puppet 数据
        puppet_data = []
        with open(puppet_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                puppet_data.append([float(x) for x in row])
        puppet_np = np.array(puppet_data)
        
        # 读取 master 数据
        master_data = []
        with open(master_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            for row in reader:
                master_data.append([float(x) for x in row])
        master_np = np.array(master_data)
        
    except Exception as e:
        print(f"读取CSV失败: {e}")
        return

    if puppet_np.size == 0 or master_np.size == 0:
        print("数据为空")
        return

    puppet_time = puppet_np[:, 0]
    puppet_joints = puppet_np[:, 1:]
    
    master_time = master_np[:, 0]
    master_joints = master_np[:, 1:]
    
    num_joints = puppet_joints.shape[1]
    
    print(f"对比 {num_joints} 个关节...")

    fig, axes = plt.subplots(num_joints, 1, sharex=True, figsize=(12, 2.5 * num_joints))
    if num_joints == 1:
        axes = [axes]
    
    for i in range(num_joints):
        ax = axes[i]
        label_name = f"joint{i}" if i < 6 else "gripper"
        
        # 绘制 master (目标) 和 puppet (实际)
        ax.plot(master_time, master_joints[:, i], 'b-', label='Master (目标)', alpha=0.7, linewidth=1.5)
        ax.plot(puppet_time, puppet_joints[:, i], 'r--', label='Puppet (实际)', alpha=0.7, linewidth=1.5)
        
        ax.set_ylabel(label_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # 前6个关节设置Y轴范围为[-pi, pi]
        if i < 6:
            ax.set_ylim(-np.pi, np.pi)
            ax.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.2)
            ax.axhline(y=-np.pi, color='gray', linestyle='--', alpha=0.2)
            
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.1)

    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title("Master (目标指令) vs Puppet (实际位置)")
    plt.tight_layout()
    
    # 生成输出文件名
    root, ext = os.path.splitext(puppet_file)
    compare_filename = f"{root}_compare.png"
        
    plt.savefig(compare_filename, dpi=150)
    print(f"对比图已保存到 {compare_filename}")


def generate_test_data(filename, duration=10, step=0.02):
    """生成测试数据"""
    print(f"生成测试数据到 {filename} (时长: {duration}s)...")
    global data_records
    
    t = 0
    data = []
    
    while t < duration:
        pos_row = []
        vel_row = []
        
        # 6个机械臂关节
        for i in range(6):
            amp = 1.0 + 0.3 * (i % 3)
            freq = 0.2 + 0.1 * i
            phase = 0.5 * i
            val = amp * math.sin(2 * math.pi * freq * t + phase)
            
            w = 2 * math.pi * freq
            vel = amp * w * math.cos(w * t + phase)
            
            pos_row.append(val)
            vel_row.append(vel)
        
        # 夹爪
        gripper_val = 0.04 * (math.sin(0.5 * t) + 1)
        gripper_vel = 0.04 * 0.5 * math.cos(0.5 * t)
        
        pos_row.append(gripper_val)
        vel_row.append(gripper_vel)
        
        rec = {
            'time': t,
            'pos': pos_row,
            'vel': vel_row
        }
        data.append(rec)
        t += step
    
    data_records = data
    save_csv(filename)
    data_records = []
    print("测试数据已生成")


def sync_record_mode(filename: str, can_port: str, sample_rate: float, sync_file: str):
    """
    同步录制模式 - 等待同步文件后开始录制，使用共享时间戳
    
    同步流程:
    1. 连接到机械臂
    2. 读取 sync_file 获取共享的 start_time
    3. 删除 sync_file（通知 main 已准备好）
    4. 使用共享的 start_time 开始录制数据
    5. 收到 SIGINT (Ctrl+C) 后保存数据并退出
    
    Args:
        filename: 输出文件名
        can_port: CAN端口
        sample_rate: 采样率 (Hz)
        sync_file: 同步文件路径 (JSON 格式，包含 start_time)
    """
    global recording, data_records, silent_mode, start_time
    import json
    
    # 同步模式默认静默
    silent_mode = True
    
    # 清空之前的数据
    data_records = []
    
    # 初始化监控器
    monitor = SDKJointMonitor(can_port=can_port)
    
    if not monitor.connect():
        print("[Monitor] 连接失败, 退出")
        # 删除同步文件通知 main 失败
        if os.path.exists(sync_file):
            os.remove(sync_file)
        return
    
    # 等待连接稳定
    time.sleep(0.3)
    
    print(f"[Monitor] 已连接, 准备录制 ({can_port}, {sample_rate}Hz)")
    print(f"[Monitor] 输出文件: {filename}")
    
    # 读取共享时间戳并删除同步文件
    shared_start_time = None
    if os.path.exists(sync_file):
        try:
            with open(sync_file, 'r') as f:
                sync_data = json.load(f)
                shared_start_time = sync_data.get("start_time")
                print(f"[Monitor] 读取共享时间戳: {shared_start_time}")
        except Exception as e:
            print(f"[Monitor] 读取同步文件失败: {e}")
        os.remove(sync_file)
        print("[Monitor] 同步完成, 开始录制...")
    else:
        print("[Monitor] 警告: 同步文件不存在, 直接开始录制")
    
    # 使用共享时间戳 (如果没有则用当前时间)
    if shared_start_time:
        start_time = shared_start_time
    else:
        start_time = time.time()
    
    # 启动录制线程 (使用自定义的 start_time)
    recording = True
    thread = threading.Thread(target=_sync_recording_thread, args=(monitor, sample_rate, start_time))
    thread.daemon = True
    thread.start()
    
    try:
        # 等待中断信号
        while recording:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[Monitor] 正在停止录制...")
        recording = False
        thread.join(timeout=2.0)
        
        # 保存数据
        if data_records:
            save_csv(filename)
            print(f"[Monitor] 录制完成! 共 {len(data_records)} 条数据, 时长 {data_records[-1]['time']:.2f}s")
        else:
            print("[Monitor] 没有数据可保存")
        
        # 断开连接
        monitor.disconnect()


def _sync_recording_thread(monitor: SDKJointMonitor, sample_rate: float, shared_start_time: float):
    """同步录制线程 - 使用共享的起始时间戳"""
    global recording, data_records
    
    dt = 1.0 / sample_rate
    sample_count = 0
    
    while recording:
        loop_start = time.time()
        
        # 读取状态
        positions, velocities = monitor.get_joint_states()
        
        # 使用共享时间戳计算相对时间
        rel_time = time.time() - shared_start_time
        
        # 保存数据
        rec = {
            'time': rel_time,
            'pos': list(positions),
            'vel': list(velocities)
        }
        
        with lock:
            data_records.append(rec)
        
        sample_count += 1
        if sample_count % int(sample_rate) == 0:  # 每秒打印一次
            print(f"[Monitor] {rel_time:.1f}s, {len(data_records)} samples")
        
        # 控制采样率
        elapsed = time.time() - loop_start
        sleep_time = max(0, dt - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MIT模式下的左臂关节状态监控工具 (无需ROS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 录制左臂数据 (默认 can_left 端口)
   python monitor_joints_left_mit.py record -f left_arm.csv

2. 指定CAN端口录制
   python monitor_joints_left_mit.py record -f left_arm.csv --can-port can0

3. 指定采样率录制 (默认30Hz)
   python monitor_joints_left_mit.py record -f left_arm.csv -r 100

4. 绘制位置数据
   python monitor_joints_left_mit.py plot -f left_arm.csv

5. 绘制速度数据
   python monitor_joints_left_mit.py plot -f left_arm_vel.csv

6. 绘制加速度数据
   python monitor_joints_left_mit.py plot -f left_arm_acc.csv

7. 绘制对比图 (master vs puppet)
   python monitor_joints_left_mit.py compare -f left_arm.csv
   python monitor_joints_left_mit.py compare -f left_arm.csv -m left_arm_master.csv

8. 生成测试数据
   python monitor_joints_left_mit.py test -f test_data.csv

======== 在推理代码中使用 ========

from monitor.monitor_joints_left_mit import init_monitor, record_step, save_recorded_data

# 初始化（静默模式）
init_monitor("can_left", silent=True)

# 在推理循环中
for step in range(max_steps):
    action = policy.get_action(obs)
    obs = env.step(action)
    
    # 记录左臂数据: action[:7] 是 master 位置, action_vel[:7] 是 master 速度
    # obs["qpos"][:7] 是 puppet 位置（速度会自动从SDK读取）
    record_step(master_action=action[:7], 
                master_velocity=action_vel[:7],  # 如果有速度信息
                puppet_qpos=obs["qpos"][:7])

# 保存数据
save_recorded_data("datas/inference_left.csv")
# 会生成 6 个文件:
#   - inference_left.csv (puppet位置)
#   - inference_left_vel.csv (puppet速度)
#   - inference_left_acc.csv (puppet加速度)
#   - inference_left_master.csv (master位置)
#   - inference_left_master_vel.csv (master速度)
#   - inference_left_master_acc.csv (master加速度)

# 绘制对比图
# python monitor_joints_left_mit.py compare -f datas/inference_left.csv
        """
    )
    
    parser.add_argument(
        'mode', 
        choices=['record', 'plot', 'test', 'compare', 'sync'], 
        help="模式: record(录制), plot(绘图), test(生成测试数据), compare(对比绘图), sync(同步录制)"
    )
    
    # 获取脚本所在目录下的 datas 文件夹
    default_data_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 
        "datas"
    )
    
    parser.add_argument(
        '--file', '-f', 
        default=os.path.join(default_data_dir, "left_arm_mit.csv"), 
        help="CSV文件路径 (默认: ./datas/left_arm_mit.csv)"
    )
    parser.add_argument(
        '--can-port', '-c',
        default="can_left",
        help="CAN端口 (默认: can_left)"
    )
    parser.add_argument(
        '--sample-rate', '-r',
        type=float,
        default=30.0,
        help="采样率 Hz (默认: 30)"
    )
    parser.add_argument(
        '--master-file', '-m',
        default=None,
        help="Master CSV文件 (compare模式, 默认: {file}_master.csv)"
    )
    parser.add_argument(
        '--silent', '-s',
        action='store_true',
        help="静默模式，不打印保存信息"
    )
    parser.add_argument(
        '--sync-file',
        default="/tmp/monitor_sync_ready",
        help="同步文件路径 (sync模式使用)"
    )
    
    args = parser.parse_args()
    
    # 设置静默模式
    if args.silent:
        silent_mode = True
    
    if args.mode == 'record':
        record_mode(args.file, args.can_port, args.sample_rate)
    elif args.mode == 'plot':
        plot_mode(args.file)
    elif args.mode == 'test':
        generate_test_data(args.file)
    elif args.mode == 'sync':
        sync_record_mode(args.file, args.can_port, args.sample_rate, args.sync_file)
    elif args.mode == 'compare':
        # 确定 master 文件
        if args.master_file:
            master_file = args.master_file
        else:
            root, ext = os.path.splitext(args.file)
            master_file = f"{root}_master{ext}"
        compare_plot_mode(args.file, master_file)