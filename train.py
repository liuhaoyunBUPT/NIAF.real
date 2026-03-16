# train.py
"""
统一训练入口
支持5种模型: beast, niaf, niaf_vel, fast, oft
通过 Hydra 配置选择模型变体，通过 arm_mode 选择单/双臂

用法:
    # 训练 niaf 模型 (默认)
    python train.py --config-name config_niaf

    # 训练 niaf_vel 模型
    python train.py --config-name config_niaf_vel vel_loss_weight=0.2 jerk_loss_weight=0

    # 训练 beast 模型
    python train.py --config-name config_beast

    # 训练 fast 模型
    python train.py --config-name config_fast

    # 训练 oft 模型
    python train.py --config-name config_oft

    # 覆盖 arm_mode
    python train.py arm_mode=left

"""
import logging
from pathlib import Path
import sys
import os
import gc
import traceback

# 设置临时目录环境变量（必须在导入wandb之前）
os.environ.setdefault('TMPDIR', '/tmp')
os.environ.setdefault('TEMP', '/tmp')
os.environ.setdefault('TMP', '/tmp')

import wandb
import hydra
import torch
import importlib
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Callback, LightningModule, seed_everything, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

# 添加项目路径
sys.path.insert(0, Path(__file__).absolute().parents[1].as_posix())

from src.utils.utils import get_last_checkpoint

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def clear_cuda_cache():
    """清理CUDA缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        for i in range(torch.cuda.device_count()):
            memory_stats = torch.cuda.memory_stats(i)
            allocated = memory_stats.get('allocated_bytes.all.current', 0) / (1024**3)
            reserved = memory_stats.get('reserved_bytes.all.current', 0) / (1024**3)
            logger.info(f"GPU {i} Memory: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


@rank_zero_only
def log_rank_0(*args, **kwargs):
    logger.info(*args, **kwargs)


def setup_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """设置回调"""
    callbacks = []
    if callbacks_cfg is not None:
        for cb in callbacks_cfg.values():
            callbacks.append(hydra.utils.instantiate(cb))
    return callbacks


def setup_logger(cfg: DictConfig, model: LightningModule):
    """设置日志器"""
    return hydra.utils.instantiate(cfg.logger)


def _is_single_device(devices) -> bool:
    """判断是否为单设备训练"""
    if devices is None:
        return True
    if isinstance(devices, int):
        return devices <= 1
    if isinstance(devices, (list, tuple)):
        return len(devices) <= 1
    return False


def configure_arm_mode(cfg: DictConfig):
    """根据 arm_mode 自动配置维度和相机"""
    arm_mode = cfg.get("arm_mode", "dual")

    with open_dict(cfg):
        if arm_mode == "left":
            cfg.act_dim = 7
            cfg.use_cam_left_wrist = True
            cfg.use_cam_right_wrist = False
            log_rank_0(f"Arm mode: {arm_mode} -> act_dim=7, cameras=[left=True, right=False]")
        elif arm_mode == "right":
            cfg.act_dim = 7
            cfg.use_cam_left_wrist = False
            cfg.use_cam_right_wrist = True
            log_rank_0(f"Arm mode: {arm_mode} -> act_dim=7, cameras=[left=False, right=True]")
        else:
            log_rank_0(f"Arm mode: dual -> act_dim={cfg.get('act_dim', 14)}")


def configure_action_stats(cfg: DictConfig):
    """根据 action_mode 自动设置 action_min/action_max（如未显式指定）"""
    action_mode = cfg.get("action_mode", "relative")
    with open_dict(cfg):
        if cfg.get("action_min") is None:
            key = f"action_min_{action_mode}"
            if key not in cfg:
                raise ValueError(f"action_stats 中未定义 {key}，请检查 action_stats 配置")
            cfg.action_min = cfg[key]
        if cfg.get("action_max") is None:
            key = f"action_max_{action_mode}"
            if key not in cfg:
                raise ValueError(f"action_stats 中未定义 {key}，请检查 action_stats 配置")
            cfg.action_max = cfg[key]
    log_rank_0(f"Action stats: mode={action_mode}, min dims={len(cfg.action_min)}, max dims={len(cfg.action_max)}")


def configure_cameras(cfg: DictConfig) -> list[str]:
    """根据相机开关过滤 rgb_obs_keys"""
    active_cameras = []
    if cfg.get("use_cam_high", True):
        active_cameras.append("rgb_static")
    if cfg.get("use_cam_left_wrist", True):
        active_cameras.append("rgb_left_wrist")
    if cfg.get("use_cam_right_wrist", True):
        active_cameras.append("rgb_right_wrist")

    if len(active_cameras) == 0:
        raise ValueError("At least one camera must be enabled!")

    # 更新模型的 rgb_obs_keys
    if hasattr(cfg, 'model') and 'rgb_obs_keys' in cfg.model:
        cfg.model.rgb_obs_keys = active_cameras

    log_rank_0(f"Active cameras: {active_cameras}")
    return active_cameras


def load_model_module(model_target: str):
    """根据模型 _target_ 动态导入模型模块"""
    module_path = model_target.rsplit(".", 1)[0]
    return importlib.import_module(module_path)


@hydra.main(version_base=None, config_path="configs", config_name="config_niaf")
def train(cfg: DictConfig) -> None:
    try:
        # 设置环境
        os.environ['HYDRA_FULL_ERROR'] = '1'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        seed_everything(cfg.seed, workers=True)
        torch.set_float32_matmul_precision('medium')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        clear_cuda_cache()

        # 自动配置 arm_mode、action_stats 和相机
        configure_arm_mode(cfg)
        configure_action_stats(cfg)
        configure_cameras(cfg)

        # 获取模型类型信息
        model_target = cfg.model.get("_target_", "unknown")
        model_class_name = model_target.rsplit(".", 1)[-1] if "." in model_target else model_target

        log_rank_0(f"\n{'='*60}")
        log_rank_0(f"Initializing training for seed {cfg.seed}")
        log_rank_0(f"Data directory: {cfg.root_data_dir}")
        log_rank_0(f"Model: {model_target}")
        log_rank_0(f"Action mode: {cfg.get('action_mode', 'N/A')}")
        log_rank_0(f"Arm mode: {cfg.get('arm_mode', 'dual')}")

        # 打印 niaf_vel 特有信息
        if cfg.get('vel_loss_weight', None) is not None:
            log_rank_0(f"Loss weights: act={cfg.get('act_loss_weight', 1.0)}, "
                       f"vel={cfg.get('vel_loss_weight', 0.0)}, "
                       f"jerk={cfg.get('jerk_loss_weight', 0.0)}")
            log_rank_0(f"FPS: {cfg.get('fps', 30)}")
        log_rank_0(f"{'='*60}\n")

        # 初始化数据模块
        datamodule = hydra.utils.instantiate(cfg.datamodule)

        # 初始化模型
        checkpoint_path = get_last_checkpoint(Path.cwd())
        if checkpoint_path is None:
            model = hydra.utils.instantiate(cfg.model)
            log_rank_0(f"Initialized {model_class_name} model from scratch")
        else:
            models_m = load_model_module(model_target)
            model_class = getattr(models_m, model_class_name)
            model = model_class.load_from_checkpoint(checkpoint_path.as_posix())
            log_rank_0(f"Loaded model from checkpoint: {checkpoint_path}")

        # 设置训练
        train_logger = setup_logger(cfg, model)

        callbacks = setup_callbacks(cfg.get("callbacks", None))
        callbacks.append(LearningRateMonitor(logging_interval="step"))

        # Trainer参数
        trainer_args = {
            **cfg.trainer,
            "logger": train_logger,
            "callbacks": callbacks,
            "benchmark": False,
            "default_root_dir": Path.cwd(),
        }

        # 单卡时移除 strategy 避免报错
        if _is_single_device(trainer_args.get("devices")):
            trainer_args.pop("strategy", None)
            trainer_args.pop("sync_batchnorm", None)
            trainer_args.pop("use_distributed_sampler", None)

        # 根据设备数量选择策略
        if cfg.trainer.devices > 1:
            trainer_args["strategy"] = "ddp_find_unused_parameters_true"
            trainer_args["sync_batchnorm"] = True
            trainer_args["use_distributed_sampler"] = True

        log_rank_0(f"Training config:\n{OmegaConf.to_yaml(cfg)}")

        clear_cuda_cache()

        # 初始化Trainer并训练
        trainer = Trainer(**trainer_args)

        try:
            trainer.fit(model, datamodule=datamodule)
        except Exception as e:
            log_rank_0("\nDetailed Error Information:")
            log_rank_0("=" * 80)
            log_rank_0(f"Error Type: {type(e).__name__}")
            log_rank_0(f"Error Message: {str(e)}")
            log_rank_0("\nFull Traceback:")
            log_rank_0(traceback.format_exc())
            log_rank_0("=" * 80)
            raise e

    except Exception as e:
        logger.error(f"\nTraining failed:")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        logger.error(f"{'='*80}")
        raise e
    finally:
        clear_cuda_cache()
        cleanup_distributed()
        if wandb.run is not None:
            wandb.finish()


def cleanup_distributed():
    """清理分布式训练资源"""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # 设置临时目录（必须最先设置）
    os.environ.setdefault('TMPDIR', '/tmp')
    os.environ.setdefault('TEMP', '/tmp')
    os.environ.setdefault('TMP', '/tmp')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["TOKENIZERS_PARALLELISM"] = 'True'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'

    sys.path.insert(0, str(Path(__file__).absolute().parents[1]))

    try:
        train()
    except Exception as e:
        logger.error(f"\nTraining script failed:")
        logger.error(f"{'='*80}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        logger.error(f"{'='*80}")
        sys.exit(1)
