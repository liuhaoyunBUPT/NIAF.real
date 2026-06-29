#!/usr/bin/env python3
# -*-coding:utf8-*-
"""
MIT (Model-based Impedance Tuning) 阻抗控制模块

该模块直接使用 Piper SDK 实现阻抗控制模式，允许同时使用位置和速度信息。

阻抗控制的优势：
1. 更快的响应速度
2. 支持位置+速度的联合控制
3. 更柔顺的运动特性
"""

import time
import threading
import numpy as np
from typing import Literal, Optional, Tuple, List
from dataclasses import dataclass, field

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    try:
        # 兼容旧版本SDK
        from piper_sdk import C_PiperInterface as C_PiperInterface_V2
    except ImportError as e:
        raise ImportError(
            f"piper_sdk 未安装或无法导入: {e}\n"
            f"请确保已安装 piper_sdk: pip install piper_sdk 或从源码安装"
        )


@dataclass
class MITControlConfig:
    """MIT 控制参数配置"""
    # CAN 端口配置
    can_port_left: str = "can0"
    can_port_right: str = "can2"
    
    # MIT 控制增益参数 (SDK建议值: kp=10, kd=0.8)
    # kp: 位置比例增益, 控制位置误差对输出力矩的影响
    # kd: 速度微分增益, 控制速度误差对输出力矩的影响
    kp: float = 10.0          # SDK建议值10, 范围[0, 500], 越大位置跟踪越紧
    kd: float = 0.8           # SDK建议值0.8, 范围[-5, 5], 越大阻尼越强
    t_ref: float = 0.0        # 目标力矩参考值, 通常设为0
    
    # 每个关节的独立增益 (可选, 如果不设置则使用统一增益)
    # 格式: [joint1, joint2, joint3, joint4, joint5, joint6]
    kp_per_joint: Optional[List[float]] = None
    kd_per_joint: Optional[List[float]] = None
    
    # 夹爪控制参数
    gripper_effort: int = 1000  # 夹爪力矩 (0.001 N/m)
    
    # 安全限制
    max_joint_vel: float = 3.0  # 最大关节速度 (rad/s)
    enable_safety_check: bool = True


class MITArmController:
    """
    单臂 MIT 阻抗控制器
    
    使用 Piper SDK 的 JointMitCtrl 接口实现阻抗控制
    """
    
    def __init__(self, can_port: str, config: MITControlConfig, side: str = "left"):
        """
        初始化 MIT 控制器
        
        Args:
            can_port: CAN 端口名称 (如 "can0", "can2")
            config: MIT 控制配置
            side: 臂的标识 ("left" 或 "right")
        """
        self.can_port = can_port
        self.config = config
        self.side = side
        self._piper: Optional[C_PiperInterface_V2] = None
        self._enabled = False
        self._lock = threading.Lock()
        
        # 初始化增益参数
        if config.kp_per_joint is not None:
            self._kp = config.kp_per_joint
        else:
            self._kp = [config.kp] * 6
            
        if config.kd_per_joint is not None:
            self._kd = config.kd_per_joint
        else:
            self._kd = [config.kd] * 6
    
    def connect(self) -> bool:
        """
        连接机械臂并初始化
        
        Returns:
            连接是否成功
        """
        try:
            print(f"[MITController-{self.side}] 正在连接 {self.can_port}...")
            self._piper = C_PiperInterface_V2(self.can_port)
            self._piper.ConnectPort()
            print(f"[MITController-{self.side}] 连接成功")
            return True
        except Exception as e:
            print(f"[MITController-{self.side}] 连接失败: {e}")
            return False
    
    def enable(self) -> bool:
        """
        使能机械臂并切换到 MIT 模式
        
        Returns:
            使能是否成功
        """
        if self._piper is None:
            print(f"[MITController-{self.side}] 错误: 请先调用 connect()")
            return False
        
        try:
            print(f"[MITController-{self.side}] 正在使能机械臂...")
            
            # 使用 EnablePiper 等待使能完成（参考官方demo）
            max_retry = 50  # 最多等待5秒
            retry_count = 0
            while not self._piper.EnablePiper():
                time.sleep(0.1)
                retry_count += 1
                if retry_count >= max_retry:
                    print(f"[MITController-{self.side}] 警告: 使能超时，继续尝试...")
                    break
            
            # 使能夹爪
            self._piper.GripperCtrl(0, self.config.gripper_effort, 0x01, 0)
            time.sleep(0.2)
            
            # 检查使能状态
            enable_status = self._piper.GetArmEnableStatus()
            all_enabled = all(enable_status)
            
            if all_enabled:
                self._enabled = True
                print(f"[MITController-{self.side}] 机械臂使能成功")
            else:
                print(f"[MITController-{self.side}] 警告: 部分电机未使能 {enable_status}")
                self._enabled = True  # 仍然尝试继续
            
            return True
            
        except Exception as e:
            print(f"[MITController-{self.side}] 使能失败: {e}")
            return False
    
    def set_mit_mode(self) -> bool:
        """
        设置机械臂为 MIT 控制模式
        
        Returns:
            设置是否成功
        """
        if self._piper is None:
            return False
        
        try:
            # 设置 MIT 模式
            # ctrl_mode=0x01: CAN指令控制模式
            # move_mode=0x04: MOVE M (MIT模式)
            # move_spd_rate_ctrl=100: 速度百分比
            # is_mit_mode=0xAD: 启用MIT模式
            self._piper.ModeCtrl(
                ctrl_mode=0x01,
                move_mode=0x04,
                move_spd_rate_ctrl=100,
                is_mit_mode=0xAD
            )
            time.sleep(0.1)
            print(f"[MITController-{self.side}] MIT 模式已设置")
            return True
        except Exception as e:
            print(f"[MITController-{self.side}] 设置 MIT 模式失败: {e}")
            return False
    
    def set_position_mode(self) -> bool:
        """
        设置机械臂为位置控制模式 (MOVE J)
        
        Returns:
            设置是否成功
        """
        if self._piper is None:
            return False
        
        try:
            self._piper.ModeCtrl(
                ctrl_mode=0x01,
                move_mode=0x01,  # MOVE J
                move_spd_rate_ctrl=100,
                is_mit_mode=0x00
            )
            time.sleep(0.1)
            print(f"[MITController-{self.side}] 位置模式已设置")
            return True
        except Exception as e:
            print(f"[MITController-{self.side}] 设置位置模式失败: {e}")
            return False
    
    def send_mit_command(
        self,
        positions: np.ndarray,
        velocities: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        torques: Optional[np.ndarray] = None
    ) -> bool:
        """
        发送 MIT 控制命令
        
        Args:
            positions: 6个关节的目标位置 (rad)
            velocities: 6个关节的目标速度 (rad/s), 如果为None则使用0
            kp: 位置增益, 如果为None则使用配置值
            kd: 速度增益, 如果为None则使用配置值
            torques: 目标力矩, 如果为None则使用0
        
        Returns:
            发送是否成功
        """
        if self._piper is None:
            return False
        
        # 处理默认值
        if velocities is None:
            velocities = np.zeros(6)
        if kp is None:
            kp = np.array(self._kp)
        if kd is None:
            kd = np.array(self._kd)
        if torques is None:
            torques = np.zeros(6)
        
        # 安全检查
        if self.config.enable_safety_check:
            velocities = np.clip(velocities, -self.config.max_joint_vel, self.config.max_joint_vel)
        
        try:
            with self._lock:
                # 先发送模式控制命令确保在MIT模式（参考官方demo，move_spd_rate_ctrl=0）
                self._piper.MotionCtrl_2(
                    ctrl_mode=0x01,
                    move_mode=0x04,
                    move_spd_rate_ctrl=0,
                    is_mit_mode=0xAD
                )
                
                # 为每个关节发送MIT控制命令
                for i in range(6):
                    motor_num = i + 1  # 电机编号从1开始
                    self._piper.JointMitCtrl(
                        motor_num=motor_num,
                        pos_ref=float(positions[i]),
                        vel_ref=float(velocities[i]),
                        kp=float(kp[i]),
                        kd=float(kd[i]),
                        t_ref=float(torques[i])
                    )
            return True
            
        except Exception as e:
            print(f"[MITController-{self.side}] 发送MIT命令失败: {e}")
            return False
    
    def send_position_command(self, positions: np.ndarray) -> bool:
        """
        发送纯位置控制命令 (使用 JointCtrl)
        
        Args:
            positions: 6个关节的目标位置 (rad)
        
        Returns:
            发送是否成功
        """
        if self._piper is None:
            return False
        
        try:
            # 转换为 0.001 度 单位
            factor = 57324.840764  # 1000 * 180 / pi
            joint_angles = [int(round(pos * factor)) for pos in positions]
            
            with self._lock:
                self._piper.ModeCtrl(
                    ctrl_mode=0x01,
                    move_mode=0x01,
                    move_spd_rate_ctrl=100,
                    is_mit_mode=0x00
                )
                self._piper.JointCtrl(*joint_angles)
            return True
            
        except Exception as e:
            print(f"[MITController-{self.side}] 发送位置命令失败: {e}")
            return False
    
    def set_gripper(self, gripper_pos: float) -> bool:
        """
        控制夹爪
        
        Args:
            gripper_pos: 夹爪位置 (米), 范围 [0, 0.1]
        
        Returns:
            发送是否成功
        """
        if self._piper is None:
            return False
        
        try:
            # 转换为 0.001mm 单位
            gripper_angle = int(round(gripper_pos * 1000 * 1000))
            gripper_angle = np.clip(gripper_angle, 0, 100000)  # 限制在有效范围 [0, 0.1m]
            
            with self._lock:
                self._piper.GripperCtrl(
                    gripper_angle=gripper_angle,
                    gripper_effort=self.config.gripper_effort,
                    gripper_code=0x01,  # 使能
                    set_zero=0x00
                )
            return True
            
        except Exception as e:
            print(f"[MITController-{self.side}] 设置夹爪失败: {e}")
            return False
    
    def get_joint_states(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取当前关节状态
        
        Returns:
            (positions, velocities, efforts): 关节位置(rad), 速度(rad/s), 力矩(Nm)
        """
        if self._piper is None:
            return np.zeros(7), np.zeros(7), np.zeros(7)
        
        try:
            # 获取关节角度 (0.001度 -> rad)
            joint_msgs = self._piper.GetArmJointMsgs()
            positions = np.array([
                joint_msgs.joint_state.joint_1 / 1000 * 0.017444,
                joint_msgs.joint_state.joint_2 / 1000 * 0.017444,
                joint_msgs.joint_state.joint_3 / 1000 * 0.017444,
                joint_msgs.joint_state.joint_4 / 1000 * 0.017444,
                joint_msgs.joint_state.joint_5 / 1000 * 0.017444,
                joint_msgs.joint_state.joint_6 / 1000 * 0.017444,
            ])
            
            # 获取夹爪位置 (0.001mm -> m)
            gripper_msgs = self._piper.GetArmGripperMsgs()
            gripper_pos = gripper_msgs.gripper_state.grippers_angle / 1000000
            positions = np.append(positions, gripper_pos)
            
            # 获取速度 (0.001 rad/s -> rad/s)
            high_spd_msgs = self._piper.GetArmHighSpdInfoMsgs()
            velocities = np.array([
                high_spd_msgs.motor_1.motor_speed / 1000,
                high_spd_msgs.motor_2.motor_speed / 1000,
                high_spd_msgs.motor_3.motor_speed / 1000,
                high_spd_msgs.motor_4.motor_speed / 1000,
                high_spd_msgs.motor_5.motor_speed / 1000,
                high_spd_msgs.motor_6.motor_speed / 1000,
                0.0  # 夹爪速度
            ])
            
            # 获取力矩 (0.001 Nm -> Nm)
            efforts = np.array([
                high_spd_msgs.motor_1.effort / 1000,
                high_spd_msgs.motor_2.effort / 1000,
                high_spd_msgs.motor_3.effort / 1000,
                high_spd_msgs.motor_4.effort / 1000,
                high_spd_msgs.motor_5.effort / 1000,
                high_spd_msgs.motor_6.effort / 1000,
                gripper_msgs.gripper_state.grippers_effort / 1000
            ])
            
            return positions, velocities, efforts
            
        except Exception as e:
            print(f"[MITController-{self.side}] 获取状态失败: {e}")
            return np.zeros(7), np.zeros(7), np.zeros(7)
    
    def disable(self):
        """失能机械臂"""
        if self._piper is not None:
            try:
                self._piper.DisableArm(7)
                self._piper.GripperCtrl(0, 0, 0x00, 0)
                self._enabled = False
                print(f"[MITController-{self.side}] 机械臂已失能")
            except Exception as e:
                print(f"[MITController-{self.side}] 失能失败: {e}")
    
    def disconnect(self):
        """断开连接"""
        self.disable()
        if self._piper is not None:
            try:
                self._piper.DisconnectPort()
                print(f"[MITController-{self.side}] 已断开连接")
            except Exception as e:
                print(f"[MITController-{self.side}] 断开连接失败: {e}")
        self._piper = None


class DualArmMITController:
    """
    双臂 MIT 阻抗控制器
    
    管理左右两个机械臂的 MIT 控制
    """
    
    def __init__(self, config: Optional[MITControlConfig] = None):
        """
        初始化双臂控制器
        
        Args:
            config: MIT 控制配置, 如果为None则使用默认配置
        """
        self.config = config or MITControlConfig()
        self.left_arm = MITArmController(
            can_port=self.config.can_port_left,
            config=self.config,
            side="left"
        )
        self.right_arm = MITArmController(
            can_port=self.config.can_port_right,
            config=self.config,
            side="right"
        )
        self._use_mit_mode = False
    
    def connect_and_enable(self) -> bool:
        """
        连接并使能双臂
        
        Returns:
            是否成功
        """
        # 连接
        if not self.left_arm.connect():
            return False
        if not self.right_arm.connect():
            return False
        
        # 使能
        if not self.left_arm.enable():
            return False
        if not self.right_arm.enable():
            return False
        
        return True
    
    def set_mit_mode(self):
        """切换到 MIT 阻抗控制模式"""
        self.left_arm.set_mit_mode()
        self.right_arm.set_mit_mode()
        self._use_mit_mode = True
        print("[DualArmMIT] 已切换到 MIT 阻抗控制模式")
    
    def move_to_reset_position(
        self, 
        left_target: np.ndarray, 
        right_target: np.ndarray,
        duration: float = 2.0,
        dt: float = 0.02
    ) -> bool:
        """
        平滑移动到复位位置
        
        Args:
            left_target: 左臂目标位置 [6个关节 + 1个夹爪]
            right_target: 右臂目标位置 [6个关节 + 1个夹爪]
            duration: 移动时间 (秒)
            dt: 控制周期 (秒)
        
        Returns:
            是否成功
        """
        try:
            # 获取当前位置
            left_pos, _, _ = self.left_arm.get_joint_states()
            right_pos, _, _ = self.right_arm.get_joint_states()
            
            left_current = left_pos[:6]  # 只取关节位置
            right_current = right_pos[:6]
            
            left_target_joints = np.array(left_target[:6])
            right_target_joints = np.array(right_target[:6])
            
            # 处理夹爪值：
            # - 模型输出范围是 0-0.1 米
            # - 但 reset_position 中的值可能是 ROS 模式下的特殊值（如 3.55...）
            # - 如果值 > 1，说明是旧格式，转换为完全张开 (0.1米)
            left_gripper_raw = left_target[6] if len(left_target) > 6 else 0.1
            right_gripper_raw = right_target[6] if len(right_target) > 6 else 0.1
            
            # 转换夹爪值到 [0, 0.1] 米范围
            left_target_gripper = 0.1 if left_gripper_raw > 1.0 else left_gripper_raw
            right_target_gripper = 0.1 if right_gripper_raw > 1.0 else right_gripper_raw
            
            # 计算步数
            steps = int(duration / dt)
            
            print(f"[DualArmMIT] 开始复位移动，目标位置...")
            print(f"  左臂关节: {left_target_joints}")
            print(f"  右臂关节: {right_target_joints}")
            print(f"  左夹爪: {left_target_gripper}m, 右夹爪: {right_target_gripper}m")
            
            for i in range(steps + 1):
                # 线性插值
                alpha = i / steps
                left_interp = left_current + alpha * (left_target_joints - left_current)
                right_interp = right_current + alpha * (right_target_joints - right_current)
                left_gripper_interp = alpha * left_target_gripper
                right_gripper_interp = alpha * right_target_gripper
                
                # 发送MIT命令
                self.left_arm.send_mit_command(left_interp)
                self.right_arm.send_mit_command(right_interp)
                
                # 逐渐打开夹爪
                self.left_arm.set_gripper(left_gripper_interp)
                self.right_arm.set_gripper(right_gripper_interp)
                
                time.sleep(dt)
            
            print("[DualArmMIT] 复位移动完成")
            return True
            
        except Exception as e:
            print(f"[DualArmMIT] 复位移动失败: {e}")
            return False
    
    def set_position_mode(self):
        """切换到位置控制模式"""
        self.left_arm.set_position_mode()
        self.right_arm.set_position_mode()
        self._use_mit_mode = False
        print("[DualArmMIT] 已切换到位置控制模式")
    
    def send_action(
        self,
        action: np.ndarray,
        velocities: Optional[np.ndarray] = None
    ) -> bool:
        """
        发送控制动作
        
        Args:
            action: 14维动作向量 [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
            velocities: 14维速度向量 (可选), 仅在MIT模式下使用
        
        Returns:
            是否成功
        """
        # 解析动作
        left_arm_pos = action[:6]
        left_gripper = action[6]
        right_arm_pos = action[7:13]
        right_gripper = action[13]
        
        if self._use_mit_mode and velocities is not None:
            # MIT 模式: 使用位置+速度
            left_arm_vel = velocities[:6]
            right_arm_vel = velocities[7:13]
            
            success_left = self.left_arm.send_mit_command(left_arm_pos, left_arm_vel)
            success_right = self.right_arm.send_mit_command(right_arm_pos, right_arm_vel)
        else:
            # 位置模式
            if self._use_mit_mode:
                success_left = self.left_arm.send_mit_command(left_arm_pos)
                success_right = self.right_arm.send_mit_command(right_arm_pos)
            else:
                success_left = self.left_arm.send_position_command(left_arm_pos)
                success_right = self.right_arm.send_position_command(right_arm_pos)
        
        # 控制夹爪
        self.left_arm.set_gripper(left_gripper)
        self.right_arm.set_gripper(right_gripper)
        
        return success_left and success_right
    
    def get_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        获取双臂状态
        
        Returns:
            (qpos, qvel, effort): 14维的位置、速度、力矩
        """
        left_pos, left_vel, left_eff = self.left_arm.get_joint_states()
        right_pos, right_vel, right_eff = self.right_arm.get_joint_states()
        
        qpos = np.concatenate([left_pos, right_pos])
        qvel = np.concatenate([left_vel, right_vel])
        effort = np.concatenate([left_eff, right_eff])
        
        return qpos, qvel, effort
    
    def disable(self):
        """失能双臂"""
        self.left_arm.disable()
        self.right_arm.disable()
    
    def disconnect(self):
        """断开连接"""
        self.left_arm.disconnect()
        self.right_arm.disconnect()


# ============================================================================
# 测试代码
# ============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="MIT 控制器测试")
    parser.add_argument("--can-left", default="can0", help="左臂CAN端口")
    parser.add_argument("--can-right", default="can2", help="右臂CAN端口")
    parser.add_argument("--kp", type=float, default=30.0, help="位置增益")
    parser.add_argument("--kd", type=float, default=1.0, help="速度增益")
    args = parser.parse_args()
    
    config = MITControlConfig(
        can_port_left=args.can_left,
        can_port_right=args.can_right,
        kp=args.kp,
        kd=args.kd
    )
    
    controller = DualArmMITController(config)
    
    try:
        print("正在连接和使能机械臂...")
        if not controller.connect_and_enable():
            print("连接或使能失败!")
            exit(1)
        
        print("切换到 MIT 模式...")
        controller.set_mit_mode()
        
        print("读取当前状态...")
        qpos, qvel, effort = controller.get_observation()
        print(f"位置: {qpos}")
        print(f"速度: {qvel}")
        print(f"力矩: {effort}")
        
        print("\n按 Ctrl+C 退出...")
        while True:
            time.sleep(0.1)
            qpos, qvel, effort = controller.get_observation()
            print(f"\r位置: {qpos[:6]}", end="", flush=True)
            
    except KeyboardInterrupt:
        print("\n正在退出...")
    finally:
        controller.disconnect()
        print("完成")
