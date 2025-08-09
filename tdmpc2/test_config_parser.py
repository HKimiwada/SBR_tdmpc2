# test_config_parser.py - Test-friendly version of config parsing
import dataclasses
import re
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
from common import MODEL_SIZE, TASK_SET


def cfg_to_dataclass(cfg, frozen=False):
    """
    Converts an OmegaConf config to a dataclass object.
    This prevents graph breaks when used with torch.compile.
    """
    cfg_dict = OmegaConf.to_container(cfg)
    fields = []
    for key, value in cfg_dict.items():
        fields.append((key, Any, dataclasses.field(default_factory=lambda value_=value: value_)))
    dataclass_name = "Config"
    dataclass = dataclasses.make_dataclass(dataclass_name, fields, frozen=frozen)
    def get(self, val, default=None):
        return getattr(self, val, default)
    dataclass.get = get
    return dataclass()


def parse_cfg_for_testing(cfg: OmegaConf, work_dir_base: str = "./test_logs") -> OmegaConf:
    """
    Test-friendly version of parse_cfg that doesn't require Hydra initialization.
    """

    # Logic
    for k in cfg.keys():
        try:
            v = cfg[k]
            if v == None:
                v = True
        except:
            pass

    # Algebraic expressions
    for k in cfg.keys():
        try:
            v = cfg[k]
            if isinstance(v, str):
                match = re.match(r"(\d+)([+\-*/])(\d+)", v)
                if match:
                    cfg[k] = eval(match.group(1) + match.group(2) + match.group(3))
                    if isinstance(cfg[k], float) and cfg[k].is_integer():
                        cfg[k] = int(cfg[k])
        except:
            pass

    # Convert numeric strings to proper types
    numeric_fields = [
        'steps', 'batch_size', 'horizon', 'num_samples', 'num_elites', 'num_pi_trajs',
        'episode_length', 'buffer_size', 'eval_episodes', 'eval_freq', 'seed',
        'model_size', 'enc_dim', 'mlp_dim', 'latent_dim', 'task_dim', 'num_q',
        'num_enc_layers', 'num_channels', 'seed_steps', 'action_dim',
        'max_episode_steps', 'num_bins'
    ]
    
    for field in numeric_fields:
        if hasattr(cfg, field) and cfg[field] is not None:
            try:
                if isinstance(cfg[field], str):
                    # Handle underscore separators in numbers (e.g., "1_500" -> 1500)
                    clean_val = str(cfg[field]).replace('_', '')
                    if clean_val.isdigit():
                        cfg[field] = int(clean_val)
                    elif '.' in clean_val:
                        cfg[field] = float(clean_val)
                elif isinstance(cfg[field], (int, float)):
                    # Already numeric, keep as is
                    pass
            except (ValueError, AttributeError):
                # Keep original value if conversion fails
                pass

    # Convenience - Use test-friendly work_dir
    cfg.work_dir = Path(work_dir_base) / cfg.task / str(cfg.seed) / cfg.exp_name
    cfg.task_title = cfg.task.replace("-", " ").title()
    cfg.bin_size = (cfg.vmax - cfg.vmin) / (cfg.num_bins-1) # Bin size for discrete regression

    # Model size
    if cfg.get('model_size', None) is not None:
        assert cfg.model_size in MODEL_SIZE.keys(), \
            f'Invalid model size {cfg.model_size}. Must be one of {list(MODEL_SIZE.keys())}'
        for k, v in MODEL_SIZE[cfg.model_size].items():
            cfg[k] = v
        if cfg.task == 'mt30' and cfg.model_size == 19:
            cfg.latent_dim = 512 # This checkpoint is slightly smaller

    # Multi-task
    cfg.multitask = cfg.task in TASK_SET.keys()
    if cfg.multitask:
        cfg.task_title = cfg.task.upper()
        # Account for slight inconsistency in task_dim for the mt30 experiments
        cfg.task_dim = 96 if cfg.task == 'mt80' or cfg.get('model_size', 5) in {1, 317} else 64
    else:
        cfg.task_dim = 0
    cfg.tasks = TASK_SET.get(cfg.task, [cfg.task])

    # Ensure episode_length is set for custom environments
    if not hasattr(cfg, 'episode_length') or cfg.episode_length is None:
        if cfg.task.startswith('stack-'):
            cfg.episode_length = 1500  # Default for stacking tasks
        else:
            cfg.episode_length = 1000  # Default fallback
    
    # Ensure episode_length is an integer
    if isinstance(cfg.episode_length, str):
        cfg.episode_length = int(cfg.episode_length.replace('_', ''))
    
    # Set other required fields that might be missing
    if not hasattr(cfg, 'seed_steps') or cfg.seed_steps is None:
        cfg.seed_steps = max(1000, 5 * cfg.episode_length)
    
    if not hasattr(cfg, 'action_dim') or cfg.action_dim is None:
        cfg.action_dim = 9  # Default for stacking environment
    
    # Ensure all required numeric fields are integers
    required_int_fields = ['episode_length', 'seed_steps', 'action_dim', 'steps', 'batch_size']
    for field in required_int_fields:
        if hasattr(cfg, field) and cfg[field] is not None:
            cfg[field] = int(cfg[field])

    return cfg_to_dataclass(cfg)