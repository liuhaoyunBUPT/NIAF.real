# Ignore lint errors because this file is mostly copied from ACT (https://github.com/tonyzhaozh/act).
# ruff: noqa
import collections
import time
import os
import csv
from typing import Optional, List, Literal
import dm_env
# from interbotix_xs_modules.arm import InterbotixManipulatorXS
# from interbotix_xs_msgs.msg import JointSingleCommand
import numpy as np

try:
    from agilex_cobot_magic import constants
    from agilex_cobot_magic import robot_utils
except ModuleNotFoundError as exc:
    if exc.name != "agilex_cobot_magic":
        raise
    import constants
    import robot_utils

try:
    try:
        from agilex_cobot_magic.monitor import monitor_joints_left_mit as _timeline_monitor
    except ModuleNotFoundError as exc:
        if exc.name != "agilex_cobot_magic":
            raise
        from monitor import monitor_joints_left_mit as _timeline_monitor
except Exception:
    _timeline_monitor = None

# MIT控制模式支持
try:
    try:
        from agilex_cobot_magic.mit_controller import (
            DualArmMITController,
            MITControlConfig,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "agilex_cobot_magic":
            raise
        from mit_controller import DualArmMITController, MITControlConfig
    MIT_AVAILABLE = True
except ImportError as e:
    MIT_AVAILABLE = False
    import traceback
    print(f"[RealEnv] 警告: MIT控制模块未找到，仅支持位置控制模式")
    print(f"[RealEnv] 导入错误详情: {e}")
    traceback.print_exc()

# This is the reset position that is used by the AgileX Cobot Magic runtime.
DEFAULT_RESET_POSITION = [0, -0.96, 1.16, 0, -0.3, 0]

# 控制模式枚举
CONTROL_MODE_POSITION = "position"  # 通过ROS话题的位置控制
CONTROL_MODE_MIT = "mit"            # 直接SDK的MIT阻抗控制


class MasterRecorder:
    """Master 指令记录器 - 记录发送给机械臂的控制指令"""
    
    def __init__(self, start_time: float, output_file: str):
        self.start_time = start_time
        self.output_file = output_file
        self.records = []
        print(f"[MasterRecorder] 初始化, start_time={start_time}, output={output_file}")
    
    def record(self, action: np.ndarray, velocity: Optional[np.ndarray] = None):
        """记录一条 master 指令"""
        rel_time = time.time() - self.start_time
        
        if velocity is None:
            velocity = np.zeros(7)
        
        self.records.append({
            'time': rel_time,
            'pos': list(action[:7]),  # 左臂 7 维
            'vel': list(velocity[:7]) if len(velocity) >= 7 else list(velocity) + [0.0] * (7 - len(velocity))
        })
    
    def save(self):
        """保存 master 数据到 CSV (位置、速度、加速度)"""
        if not self.records:
            print("[MasterRecorder] 没有数据可保存")
            return
        
        # 构建 master 文件名
        root, ext = os.path.splitext(self.output_file)
        master_file = f"{root}_master{ext}"
        
        times = [r['time'] for r in self.records]
        positions = [r['pos'] for r in self.records]
        velocities = [r['vel'] for r in self.records]
        
        header = ["time"] + [f"joint{i}" for i in range(len(positions[0]))]
        
        # 1. 保存位置
        self._write_csv(master_file, times, positions, header)
        
        # 2. 保存速度
        vel_file = f"{root}_master_vel{ext}"
        self._write_csv(vel_file, times, velocities, header)
        
        # 3. 保存加速度 (速度的导数)
        accelerations = self._calc_derivative(times, velocities)
        acc_file = f"{root}_master_acc{ext}"
        self._write_csv(acc_file, times, accelerations, header)
        
        print(f"[MasterRecorder] 已保存 {len(self.records)} 条记录到 {master_file}")
    
    def _write_csv(self, filename: str, times: list, data: list, header: list):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for t, row in zip(times, data):
                writer.writerow([t] + row)
    
    def _calc_derivative(self, times: list, values: list) -> list:
        if len(times) < 2:
            return [[0.0] * len(values[0])] * len(values) if values else []
        derivs = [[0.0] * len(values[0])]
        for i in range(1, len(times)):
            dt = times[i] - times[i-1]
            row = [(values[i][j] - values[i-1][j]) / dt if dt > 1e-6 else 0.0 
                   for j in range(len(values[0]))]
            derivs.append(row)
        return derivs


class RealEnv:
    """
    Environment for real robot bi-manual manipulation
    
    支持两种控制模式:
    1. position: 通过ROS话题的位置控制 (默认)
    2. mit: 直接SDK的MIT阻抗控制 (支持位置+速度)
    
    Action space:      [left_arm_qpos (6),             # absolute joint position
                        left_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
                        right_arm_qpos (6),            # absolute joint position
                        right_gripper_positions (1),]  # normalized gripper position (0: close, 1: open)

    Observation space: {"qpos": Concat[ left_arm_qpos (6),          # absolute joint position
                                        left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
                                        right_arm_qpos (6),         # absolute joint position
                                        right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
                        "qvel": Concat[ left_arm_qvel (6),         # absolute joint velocity (rad)
                                        left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
                                        right_arm_qvel (6),         # absolute joint velocity (rad)
                                        right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
                        "images": {"cam_high": (480x640x3),        # h, w, c, dtype='uint8'
                                   "cam_low": (480x640x3),         # h, w, c, dtype='uint8'
                                   "cam_left_wrist": (480x640x3),  # h, w, c, dtype='uint8'
                                   "cam_right_wrist": (480x640x3)} # h, w, c, dtype='uint8'
    """

    def __init__(
        self, 
        init_node, 
        *, 
        reset_position: Optional[List[float]] = None, 
        setup_robots: bool = True,
        control_mode: str = CONTROL_MODE_POSITION,
        mit_config: Optional[dict] = None,
        record_config: Optional[dict] = None,
    ):
        """
        初始化 RealEnv
        
        Args:
            init_node: 是否初始化ROS节点
            reset_position: 复位位置
            setup_robots: 是否设置机器人
            control_mode: 控制模式 ("position" 或 "mit")
            mit_config: MIT控制配置字典，包含:
                - can_port_left: 左臂CAN端口 (默认 "can0")
                - can_port_right: 右臂CAN端口 (默认 "can2")
                - kp: 位置增益 (默认 30.0)
                - kd: 速度增益 (默认 1.0)
        """
        # reset_position = START_ARM_POSE[:6]
        self._reset_position = reset_position[:6] if reset_position else DEFAULT_RESET_POSITION
        self._reset_position_left0= [-0.00133514404296875, 0.00209808349609375, 0.01583099365234375, -0.032616615295410156, -0.00286102294921875, 0.00095367431640625, 3.557830810546875]
        self._reset_position_right0 = [-0.00133514404296875, 0.00438690185546875, 0.034523963928222656, -0.053597450256347656, -0.00476837158203125, -0.00209808349609375, 3.557830810546875]

        # 控制模式
        self._control_mode = control_mode
        self._mit_controller = None
        self._record_config = record_config or {}
        self._timeline_recording_enabled = False
        
        # 根据控制模式初始化
        if control_mode == CONTROL_MODE_MIT:
            if not MIT_AVAILABLE:
                raise RuntimeError("MIT控制模块不可用，请检查 mit_controller.py 是否存在")
            
            # 创建MIT配置
            mit_cfg = mit_config or {}
            config = MITControlConfig(
                can_port_left=mit_cfg.get("can_port_left", "can0"),
                can_port_right=mit_cfg.get("can_port_right", "can2"),
                kp=mit_cfg.get("kp", 30.0),
                kd=mit_cfg.get("kd", 1.0),
                gripper_effort=mit_cfg.get("gripper_effort", 1000),
            )
            
            # 初始化MIT控制器
            self._mit_controller = DualArmMITController(config)
            if not self._mit_controller.connect_and_enable():
                raise RuntimeError("MIT控制器连接或使能失败")
            self._mit_controller.set_mit_mode()
            print(f"[RealEnv] MIT阻抗控制模式已启用 (kp={config.kp}, kd={config.kd})")
        else:
            print(f"[RealEnv] 位置控制模式已启用")
        
        # 时间轴记录器（MIT模式）：结束后直接绘图，不走 CSV 主流程
        if (
            self._control_mode == CONTROL_MODE_MIT
            and self._record_config.get("enabled")
            and self._record_config.get("start_time") is not None
            and self._record_config.get("output_file")
            and _timeline_monitor is not None
        ):
            can_port_left = "can_left"
            if mit_config is not None:
                can_port_left = mit_config.get("can_port_left", can_port_left)
            ok = _timeline_monitor.init_monitor(
                can_port=can_port_left,
                silent=True,
                start_time=float(self._record_config["start_time"]),
            )
            if ok:
                _timeline_monitor.init_timeline_recorder(self._record_config["output_file"])
                self._timeline_recording_enabled = True
                print("[RealEnv] 时间轴可视化记录已启用")
            else:
                print("[RealEnv] 时间轴可视化记录初始化失败")

        self.args = robot_utils.get_arguments()
        self.ros_operator = robot_utils.RosOperator(self.args)

        # self.puppet_bot_left = InterbotixManipulatorXS(
        #     robot_model="vx300s",
        #     group_name="arm",
        #     gripper_name="gripper",
        #     robot_name="puppet_left",
        #     init_node=init_node,
        # )
        # self.puppet_bot_right = InterbotixManipulatorXS(
        #     robot_model="vx300s", group_name="arm", gripper_name="gripper", robot_name="puppet_right", init_node=False
        # )
        if setup_robots:
            self.setup_robots()

        # self.recorder_left = robot_utils.Recorder("left", init_node=False)
        # self.recorder_right = robot_utils.Recorder("right", init_node=False)
        # self.image_recorder = robot_utils.ImageRecorder(init_node=False)
        # self.gripper_command = JointSingleCommand(name="gripper")

    def setup_robots(self):
        return 0
    
    # def get_qpos(self):
    #     left_qpos_raw = self.recorder_left.qpos
    #     right_qpos_raw = self.recorder_right.qpos
    #     left_arm_qpos = left_qpos_raw[:6]
    #     right_arm_qpos = right_qpos_raw[:6]
    #     left_gripper_qpos = [
    #         constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(left_qpos_raw[7])
    #     ]  # this is position not joint
    #     right_gripper_qpos = [
    #         constants.PUPPET_GRIPPER_POSITION_NORMALIZE_FN(right_qpos_raw[7])
    #     ]  # this is position not joint
    #     return np.concatenate([left_arm_qpos, left_gripper_qpos, right_arm_qpos, right_gripper_qpos])

    # def get_qvel(self):
    #     left_qvel_raw = self.recorder_left.qvel
    #     right_qvel_raw = self.recorder_right.qvel
    #     left_arm_qvel = left_qvel_raw[:6]
    #     right_arm_qvel = right_qvel_raw[:6]
    #     left_gripper_qvel = [constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(left_qvel_raw[7])]
    #     right_gripper_qvel = [constants.PUPPET_GRIPPER_VELOCITY_NORMALIZE_FN(right_qvel_raw[7])]
    #     return np.concatenate([left_arm_qvel, left_gripper_qvel, right_arm_qvel, right_gripper_qvel])

        # def get_effort(self):
        #     left_effort_raw = self.recorder_left.effort
        #     right_effort_raw = self.recorder_right.effort
        #     left_robot_effort = left_effort_raw[:7]
        #     right_robot_effort = right_effort_raw[:7]
        #     return np.concatenate([left_robot_effort, right_robot_effort])

    # def get_images(self):
    #     return self.image_recorder.get_images()



    def build_image_dict(self,img_front: np.ndarray,
                        img_left:  np.ndarray,
                        img_right: np.ndarray) -> dict[str, np.ndarray | None]:
        """将三路 RGB 帧封装成 ImageRecorder.get_images 同格式 dict。"""
        return {
            "cam_high":           img_front,
            "cam_high_depth":     None,          # 若无深度帧可置 None
            "cam_left_wrist":     img_left,
            "cam_left_wrist_depth":  None,
            "cam_right_wrist":    img_right,
            "cam_right_wrist_depth": None,
            }

    # def set_gripper_pose(self, left_gripper_desired_pos_normalized, right_gripper_desired_pos_normalized):
    #     left_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_gripper_desired_pos_normalized)
    #     self.gripper_command.cmd = left_gripper_desired_joint
    #     self.puppet_bot_left.gripper.core.pub_single.publish(self.gripper_command)

    #     right_gripper_desired_joint = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(
    #         right_gripper_desired_pos_normalized
    #     )
    #     self.gripper_command.cmd = right_gripper_desired_joint
    #     self.puppet_bot_right.gripper.core.pub_single.publish(self.gripper_command)

    # def _reset_joints(self):
    #     robot_utils.move_arms(
    #         [self.puppet_bot_left, self.puppet_bot_right], [self._reset_position, self._reset_position], move_time=1
    #     )

    # def _reset_gripper(self):
    #     """Set to position mode and do position resets: first open then close. Then change back to PWM mode"""
    #     robot_utils.move_grippers(
    #         [self.puppet_bot_left, self.puppet_bot_right], [constants.PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5
    #     )
    #     robot_utils.move_grippers(
    #         [self.puppet_bot_left, self.puppet_bot_right], [constants.PUPPET_GRIPPER_JOINT_CLOSE] * 2, move_time=1
    #     )

    # def get_observation(self):
    #     obs = collections.OrderedDict()
    #     obs["qpos"] = self.get_qpos()
    #     obs["qvel"] = self.get_qvel()
    #     obs["effort"] = self.get_effort()
    #     obs["images"] = self.get_images()
    #     return obs

    def get_observation(self):
        """
        获取一帧同步观测并封装成 OrderedDict。
        
        MIT模式：关节状态从SDK获取，图像从ROS获取
        位置模式：全部从ROS获取
        """
        if self._control_mode == CONTROL_MODE_MIT and self._mit_controller is not None:
            # MIT模式：关节状态从SDK获取，图像从ROS获取
            (img_front, img_left, img_right) = robot_utils.get_ros_images_only(
                self.args, self.ros_operator
            )
            
            # 从MIT控制器获取关节状态
            qpos, qvel, effort = self._mit_controller.get_observation()
            
            print(f"[MIT] img_front:shape={img_front.shape}, qpos[:7]={qpos[:7]}")
        else:
            # 位置模式：全部从ROS获取
            (img_front, img_left, img_right,
             puppet_arm_left, puppet_arm_right) = robot_utils.get_ros_observation(
                self.args, self.ros_operator
            )
            print(f"img_front:shape={img_front.shape},ftype={img_front.dtype},min={img_front.min()},max={img_front.max()}")
           
            # --- 关节状态 ----------------------------------------------------------
            qpos   = np.concatenate(
                (np.asarray(puppet_arm_left.position),
                 np.asarray(puppet_arm_right.position)),
                axis=0
            )                             # shape = (14,)
            qvel   = np.concatenate(
                (np.asarray(puppet_arm_left.velocity),
                 np.asarray(puppet_arm_right.velocity)),
                axis=0
            )
            effort = np.concatenate(
                (np.asarray(puppet_arm_left.effort),
                 np.asarray(puppet_arm_right.effort)),
                axis=0
            )

        # --- 图像 --------------------------------------------------------------
        images = self.build_image_dict(img_front, img_left, img_right)
        
        # --- 打包成 OrderedDict -------------------------------------------------
        obs = collections.OrderedDict()
        obs["qpos"]   = qpos
        obs["qvel"]   = qvel
        obs["effort"] = effort
        obs["images"] = images
    
        return obs
    
    def get_reward(self):
        return 0

    # def reset(self, *, fake=False):
    #     if not fake:
    #         # Reboot puppet robot gripper motors
    #         self.puppet_bot_left.dxl.robot_reboot_motors("single", "gripper", True)
    #         self.puppet_bot_right.dxl.robot_reboot_motors("single", "gripper", True)
    #         self._reset_joints()
    #         self._reset_gripper()
    #     return dm_env.TimeStep(
    #         step_type=dm_env.StepType.FIRST, reward=self.get_reward(), discount=None, observation=self.get_observation()
    #     )

    def reset(self, *, fake: bool = False):
        """
        复位环境：
            1. 真实模式下重新上电并复位 gripper
            2. 将双臂平滑移动到 _reset_position_left0 / _reset_position_right0
            3. 返回 dm_env 的 FIRST TimeStep
        """
        if not fake:
            # 根据控制模式选择复位方式
            if self._control_mode == CONTROL_MODE_MIT and self._mit_controller is not None:
                # MIT模式：使用MIT控制器进行复位
                print("[RealEnv] MIT模式：使用MIT控制器复位...")
                self._mit_controller.move_to_reset_position(
                    np.array(self._reset_position_left0),
                    np.array(self._reset_position_right0),
                    duration=2.0
                )
            else:
                # 位置模式：使用ROS话题复位
                self.ros_operator.puppet_arm_publish_continuous(
                    self._reset_position_left0,
                    self._reset_position_right0
                )

        # 给学习框架返回 FIRST 时间步
        return dm_env.TimeStep(
            step_type  = dm_env.StepType.FIRST,
            reward     = self.get_reward(),
            discount   = None,
            observation= self.get_observation()
        )


    # def step(self, action):
    #     state_len = int(len(action) / 2)
    #     left_action = action[:state_len]
    #     right_action = action[state_len:]
    #     # self.puppet_bot_left.arm.set_joint_positions(left_action[:6], blocking=False)
    #     # self.puppet_bot_right.arm.set_joint_positions(right_action[:6], blocking=False)
    #     # self.set_gripper_pose(left_action[-1], right_action[-1])
        
    #     time.sleep(constants.DT)
    #     return dm_env.TimeStep(
    #         step_type=dm_env.StepType.MID, reward=self.get_reward(), discount=None, observation=self.get_observation()
    #     )

    def step(
        self,
        action,
        velocities: Optional[np.ndarray] = None,
        chunk_id: int = -1,
        in_overlap: bool = False,
        chunk_meta: Optional[dict] = None,
    ):
        """
        执行一步控制
        
        Args:
            action: 14维位置动作向量 [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]
            velocities: 14维速度向量 (可选), 仅在MIT模式下使用
        """
        # 1) 拆左右 7 维动作 ----------------------------------------------------------------
        state_len   = len(action) // 2
        left_action = action[:state_len]         # [arm6, grip_norm]
        right_action = action[state_len:]        # [arm6, grip_norm]

        print("[STEP] raw  action :", [round(x, 3) for x in action])

        # 2) 反归一化夹爪值 -----------------------------------------------------------------
        left_arm_target  = np.array(left_action,  dtype=float)
        right_arm_target = np.array(right_action, dtype=float)

        # left_arm_target[-1]  = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(left_arm_target[-1])
        # right_arm_target[-1] = constants.PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(right_arm_target[-1])

        print("[STEP] left target :", [round(x, 3) for x in left_arm_target])
        print("[STEP] right target:", [round(x, 3) for x in right_arm_target])

        # 3) 根据控制模式发送命令 -------------------------------------------------------------
        if self._control_mode == CONTROL_MODE_MIT and self._mit_controller is not None:
            # MIT 阻抗控制模式
            try:
                self._mit_controller.send_action(action, velocities)
                print("[STEP] MIT control OK")
            except Exception as e:
                print("[STEP] ERROR MIT control:", e)
                raise
        else:
            # 位置控制模式 (通过ROS话题)
            try:
                self.ros_operator.puppet_arm_publish(
                    left_arm_target.tolist(),
                    right_arm_target.tolist()
                )
                print("[STEP] publish OK")
            except Exception as e:
                print("[STEP] ERROR publish:", e)
                raise

        # 4) 控制周期同步 -------------------------------------------------------------------
        time.sleep(constants.DT)

        # 5) 获取新观测并记录时间轴数据
        obs = self.get_observation()
        if self._timeline_recording_enabled and self._control_mode == CONTROL_MODE_MIT and _timeline_monitor is not None:
            try:
                model_pos7 = np.asarray(left_action, dtype=np.float32)
                if velocities is not None and len(velocities) >= 7:
                    model_vel7 = np.asarray(velocities[:7], dtype=np.float32)
                else:
                    model_vel7 = np.zeros(7, dtype=np.float32)

                exec_qpos7 = np.asarray(obs["qpos"][:7], dtype=np.float32)
                exec_qvel7 = np.asarray(obs["qvel"][:7], dtype=np.float32)

                _timeline_monitor.record_timeline_step(
                    dt=float(1.0 / max(1.0, float(getattr(self.args, "ctrl_freq", 30.0)))),
                    model_pos7=model_pos7,
                    model_vel7=model_vel7,
                    exec_qpos7=exec_qpos7,
                    exec_qvel7=exec_qvel7,
                    chunk_id=int(chunk_id),
                    in_overlap=bool(in_overlap),
                    chunk_step=int((chunk_meta or {}).get("chunk_step", -1)),
                    chunk_actions_full=(chunk_meta or {}).get("chunk_actions_full"),
                    chunk_velocities_full=(chunk_meta or {}).get("chunk_velocities_full"),
                    chunk_skip_steps=int((chunk_meta or {}).get("chunk_skip_steps", 0)),
                    chunk_blend_window=int((chunk_meta or {}).get("chunk_blend_window", 0)),
                )
            except Exception as e:
                print(f"[RealEnv] 记录时间轴失败: {e}")

        # 6) 返回新的 dm_env.TimeStep
        return dm_env.TimeStep(
                step_type  = dm_env.StepType.MID,
                reward     = self.get_reward(),
                discount   = None,
                observation= obs
        )
    
    def close(self):
        """关闭环境，释放资源"""
        if self._timeline_recording_enabled and _timeline_monitor is not None:
            try:
                _timeline_monitor.save_timeline_plots()
                _timeline_monitor.shutdown_monitor()
            except Exception as e:
                print(f"[RealEnv] 保存时间轴图失败: {e}")
            self._timeline_recording_enabled = False
        
        if self._mit_controller is not None:
            self._mit_controller.disconnect()
            self._mit_controller = None
            print("[RealEnv] MIT控制器已关闭")


def get_action(master_bot_left, master_bot_right):
    action = np.zeros(14)  # 6 joint + 1 gripper, for two arms
    # Arm actions
    action[:6] = master_bot_left.dxl.joint_states.position[:6]
    action[7 : 7 + 6] = master_bot_right.dxl.joint_states.position[:6]
    # Gripper actions
    # action[6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_left.dxl.joint_states.position[6])
    # action[7 + 6] = constants.MASTER_GRIPPER_JOINT_NORMALIZE_FN(master_bot_right.dxl.joint_states.position[6])
    action[6] = master_bot_left.dxl.joint_states.position[6]
    action[7 + 6] = master_bot_right.dxl.joint_states.position[6]

    return action


def make_real_env(
    init_node, 
    *, 
    reset_position: Optional[List[float]] = None, 
    setup_robots: bool = True,
    control_mode: str = CONTROL_MODE_POSITION,
    mit_config: Optional[dict] = None,
    record_config: Optional[dict] = None,
) -> RealEnv:
    """
    创建真实环境
    
    Args:
        init_node: 是否初始化ROS节点
        reset_position: 复位位置
        setup_robots: 是否设置机器人
        control_mode: 控制模式 ("position" 或 "mit")
        mit_config: MIT控制配置
    
    Returns:
        RealEnv 实例
    """
    return RealEnv(
        init_node, 
        reset_position=reset_position, 
        setup_robots=setup_robots,
        control_mode=control_mode,
        mit_config=mit_config,
        record_config=record_config,
    )
