"""
AgileX Cobot Magic - Client Main Entry (Asynchronous Version)

【启动命令示例】
# 推荐的标准真机运行命令 (MIT模式 + 异步推理)：
python main.py --control-mode mit --async-infer --trigger-step 15 --blend-window 10 --host 10.22.19.4 --port 8000 --record

【核心启动参数说明】
- --host / --port : VLA 模型服务器 (NIAF) 的 IP 和端口，默认 10.22.19.4:8000。
- --control-mode  : 控制模式，可选 'mit' 或 'position'。真机强烈建议使用 mit。
- --async-infer   : [核心] 开启异步推理，允许机械臂执行动作的同时间后台请求下一帧网络数据。
- --trigger-step  : (仅异步有效) 当执行到当前 chunk 的第几步时，触发下一轮预测网络请求 (默认25)。
- --blend-window  : (仅异步有效) 新老 chunk 切换时，修正真实位置与预测位置跳变的平滑过渡期 (默认5)。
- --record        : 开启数据录制，会自动启动 monitor 子进程持续保存系统运行状态。
- --record-file   : 指定数据保存的文件名（无需后缀），为空时会自动生成当前时间戳。

【异步实现原理】
通过 AsyncActionChunkBroker 将推理与执行双线程解耦：
1. 主线程维持按 30Hz 稳态提取并发送动作目标给环境组件执行。
2. 当执行计数 `k_exec` 达到 `trigger_step` 时，在后台线程将当期状态发给 Server 获取下一个 Action Chunk。
3. 后台拿到结果后，比较当前机械臂的真实物理位置 与 新预测开头动作 之间的位置误差 (Gap)。
4. 使用 Smoothstep(平滑阶跃函数) 对新预测的开端位置进行动态补偿，在 `blend_window` 步内将偏差平滑衰减至0。
"""

import dataclasses
import logging
import os
import time

try:
    from agilex_cobot_magic import action_chunk_broker
    from agilex_cobot_magic import async_action_chunk_broker
except ModuleNotFoundError as exc:
    if exc.name != "agilex_cobot_magic":
        raise
    import action_chunk_broker
    import async_action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

def _load_environment_module():
    try:
        from agilex_cobot_magic import env as env_module
        return env_module
    except ModuleNotFoundError as exc:
        if exc.name != "agilex_cobot_magic":
            raise
        import env as env_module
        return env_module


MONITOR_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "monitor", "datas"
)

# 全局记录配置 (供 real_env.py 读取)
RECORD_CONFIG = {
    "enabled": False,
    "start_time": None,
    "output_file": None,
}


@dataclasses.dataclass
class Args:
    host: str = "10.22.19.4"
    port: int = 8000
    action_horizon: int = 50

    num_episodes: int = 1
    max_episode_steps: int = 10000
    
    # 控制模式配置
    control_mode: str = "position"  # "position" 或 "mit"
    
    # MIT 控制参数 (SDK建议值: kp=10, kd=0.8)
    can_port_left: str = "can_left"    # 左臂CAN端口
    can_port_right: str = "can_right"  # 右臂CAN端口
    mit_kp: float = 10.0  # 位置增益，SDK建议值10，范围[0, 500]
    mit_kd: float = 0.8   # 速度增益，SDK建议值0.8，范围[-5, 5]
    
    # 异步推理配置
    async_infer: bool = False  # 是否使用异步推理执行 (True: 推理执行重叠; False: 同步)
    trigger_step: int = 25    # 执行到第几步时触发后台推理 (建议 action_horizon 的一半)
    blend_window: int = 5     # chunk切换时的过渡窗口步数 (smoothstep平滑修正)
    
    # 关节数据记录 (仅MIT模式有效)
    record: bool = False  # 是否同步启动 monitor 记录关节数据
    record_rate: float = 50.0  # monitor 采样率 (Hz)
    record_file: str = ""  # 保存文件名 (不含路径和后缀)，为空则自动用时间戳命名


def main(args: Args) -> None:
    record_config = _prepare_record_config(args)
    
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    metadata = ws_client_policy.get_server_metadata()
    
    # 从服务器 metadata 获取模型类型，用于区分图像处理分支
    model_type = metadata.get("model_type", "pi")  # 默认为 pi 保持后向兼容
    logging.info(f"Detected model type from server: {model_type}")
    
    # 构建MIT配置
    mit_config = None
    if args.control_mode == "mit":
        mit_config = {
            "can_port_left": args.can_port_left,
            "can_port_right": args.can_port_right,
            "kp": args.mit_kp,
            "kd": args.mit_kd,
        }
        logging.info(f"MIT control mode enabled with config: {mit_config}")
    
    # 根据 async_infer 选择同步/异步 Broker
    if args.async_infer:
        broker = async_action_chunk_broker.AsyncActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.action_horizon,
            trigger_step=args.trigger_step,
            blend_window=args.blend_window,
        )
        logging.info(f"使用异步推理 Broker: trigger_step={args.trigger_step}, "
                     f"blend_window={args.blend_window}")
    else:
        broker = action_chunk_broker.ActionChunkBroker(
            policy=ws_client_policy,
            action_horizon=args.action_horizon,
            chunk_sleep=0,
        )
        logging.info("使用同步推理 Broker")
    
    env_module = _load_environment_module()
    environment = env_module.AgileXCobotMagicEnvironment(
        reset_position=metadata.get("reset_pose"),
        model_type=model_type,
        control_mode=args.control_mode,
        mit_config=mit_config,
        record_config=record_config,
    )

    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(policy=broker),
        subscribers=[],
        max_hz=30,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    try:
        runtime.run()
    finally:
        try:
            environment.close()
        except Exception as e:
            logging.error(f"关闭环境失败: {e}")


def _prepare_record_config(args: Args) -> dict:
    """准备记录配置：MIT模式下启用内存记录，结束后直接绘图。"""
    from datetime import datetime
    global RECORD_CONFIG

    RECORD_CONFIG["enabled"] = False
    RECORD_CONFIG["start_time"] = None
    RECORD_CONFIG["output_file"] = None

    if not (args.record and args.control_mode == "mit"):
        return dict(RECORD_CONFIG)

    if args.record_file:
        output_file = os.path.join(MONITOR_OUTPUT_DIR, f"{args.record_file}.csv")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(MONITOR_OUTPUT_DIR, f"inference_left_{timestamp}.csv")

    os.makedirs(MONITOR_OUTPUT_DIR, exist_ok=True)
    start_time = time.time()

    RECORD_CONFIG["enabled"] = True
    RECORD_CONFIG["start_time"] = start_time
    RECORD_CONFIG["output_file"] = output_file
    logging.info(f"启用时间轴记录，输出前缀: {output_file}")

    return dict(RECORD_CONFIG)


if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    main(args)

# python -m agilex_cobot_magic.main.py
# python main.py --control-mode mit --async-infer
