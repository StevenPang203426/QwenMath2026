"""
日志与实验追踪模块
集成 wandb 进行实验管理
"""
import logging
from typing import Optional

logger = logging.getLogger("math_solver")


def setup_logging(level: str = "INFO") -> None:
    """配置全局日志格式"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_wandb(
    project: str = "math-solver",
    run_name: Optional[str] = None,
    config: Optional[dict] = None,
    tags: Optional[list[str]] = None,
) -> None:
    """
    初始化 wandb 实验追踪

    Args:
        project: wandb 项目名称
        run_name: 实验名称
        config: 超参配置字典
        tags: 实验标签
    """
    try:
        import wandb
        wandb.init(
            project=project,
            name=run_name,
            config=config,
            tags=tags,
            reinit=True,
        )
        logger.info(f"wandb 初始化成功: project={project}, run={run_name}")
    except ImportError:
        logger.warning("wandb 未安装，跳过实验追踪。安装: pip install wandb")
    except Exception as e:
        logger.warning(f"wandb 初始化失败: {e}，继续不追踪实验")


def log_metrics(metrics: dict, step: Optional[int] = None) -> None:
    """记录指标到 wandb"""
    try:
        import wandb
        if wandb.run is not None:
            wandb.log(metrics, step=step)
    except (ImportError, Exception):
        pass


def finish_wandb() -> None:
    """结束 wandb 运行"""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except (ImportError, Exception):
        pass
