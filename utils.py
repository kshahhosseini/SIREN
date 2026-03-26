import logging
import os

from omegaconf import OmegaConf


def load_config(config_name: str) -> OmegaConf:
    """Load the configurations.

    Args:
        config_name:
            The name of the configuration file w or w/o extension.

    Returns:
        config:
            Dictionary containing the configurations.
    """
    project_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(project_dir, "config")
    config = OmegaConf.load(
        os.path.join(config_dir, config_name),
    )

    base_conf = OmegaConf.load(
        os.path.join(config_dir, "base.yaml"),
    )
    # merge all configs.
    config = OmegaConf.merge(base_conf, config)

    if config.out_dir is None:
        config.out_dir = os.path.join(project_dir, "outputs")

    config.log_dir = os.path.join(config.out_dir, "log")
    config.log_path = os.path.join(config.log_dir, "log.log")
    config.model_dir = os.path.join(config.out_dir, "model")
    config.data_path = os.path.join(project_dir, "data", config.data_name)
    config.summaries = os.path.join(config.out_dir, "summaries")
    config = OmegaConf.to_container(
        cfg=config, resolve=True
    )  # recursively resolve and cast to dict since submodules expect dict as
    # input
    return config


def setup_logging(
    path: str,
    level: str = "INFO",
) -> logging.Logger:
    """Sets up a logger.

    Args:
        path:
            The path of the log file.
        level:
            The logging level. Defaults to "INFO".

    Returns:
        logger:
            The created logger.
    """
    # Create a logger
    datefmt = "%m/%d/%Y %I:%M:%S %p"
    logging.basicConfig(
        filename=path,
        format="%(asctime)s %(levelname)s:%(message)s",
        level=level,
        datefmt=datefmt,
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(level)

    file_handler = logging.FileHandler(path)
    file_handler.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)

    # Create a Formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt=datefmt,
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.propagate = False

    return logger


def ensure_number(
    num: int | float,
    min_val: int | float | None = None,
    max_val: int | float | None = None,
    min_inclusive: bool = True,
    max_inclusive: bool = True,
) -> int | float:
    """Ensure a number fits within a range.

    Args:
        num:
            The input number.
        min_val:
            The lower bound of number. Defaults to None meaning inf.
        max_val:
            The upper bound of number. Defaults to None meaning inf.
        min_inclusive:
            If True, the check is: number >= min_val, otherwise number >
            min_val. Defaults to True.
        max_inclusive:
            If True, the check is: number <= max_val, otherwise number <
            max_val. Defaults to True.

    Returns:
        The input number if it passes all checks.
    """
    if min_val is not None:
        is_valid = num >= min_val if min_inclusive else num > min_val
        if not is_valid:
            raise ValueError(f"number {num} is not in range.")

    if max_val is not None:
        is_valid = num <= max_val if max_inclusive else num < max_val
        if not is_valid:
            raise ValueError(f"number {num} is not in range.")

    return num
