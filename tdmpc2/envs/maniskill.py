import gymnasium as gym
import numpy as np
from envs.wrappers.timeout import Timeout  # project-local timeout wrapper

# Ensure ManiSkill2 is registered
import mani_skill2.envs  # noqa: F401

# Gymnasium TimeLimit (fallback to old Gym if needed)
try:
    from gymnasium.wrappers import TimeLimit
except Exception:  # pragma: no cover
    from gym.wrappers import TimeLimit


# -----------------------------
#  Supported ManiSkill2 tasks
# -----------------------------
MANISKILL_TASKS = {
    "lift-cube": dict(
        env="LiftCube-v0",
        control_mode="pd_ee_delta_pos",
    ),
    "pick-cube": dict(
        env="PickCube-v0",
        control_mode="pd_ee_delta_pos",
    ),
    "stack-cube": dict(
        env="StackCube-v0",
        control_mode="pd_ee_delta_pos",
    ),
    "pick-ycb": dict(
        env="PickSingleYCB-v0",
        control_mode="pd_ee_delta_pose",
    ),
    "turn-faucet": dict(
        env="TurnFaucet-v0",
        control_mode="pd_ee_delta_pose",
    ),
}


class ManiSkillWrapper(gym.Wrapper):
    """
    Bridges Gymnasium's 5-tuple API to the 4-tuple expected by TD-MPC2
    and applies action repeat. Also normalizes action space bounds to be
    uniform across dims (min of lows, max of highs).
    """

    def __init__(self, env, cfg, action_repeat: int = 2):
        super().__init__(env)
        self.env = env
        self.cfg = cfg
        self.action_repeat = int(action_repeat)

        # Pass through observation space unmodified
        self.observation_space = self.env.observation_space

        # Create a uniform Box action space (same min/max for all dims)
        low_val = float(np.min(self.env.action_space.low))
        high_val = float(np.max(self.env.action_space.high))
        self.action_space = gym.spaces.Box(
            low=np.full(self.env.action_space.shape, low_val, dtype=np.float32),
            high=np.full(self.env.action_space.shape, high_val, dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, **kwargs):
        """
        Gymnasium returns (obs, info); older Gym returns obs.
        TD-MPC2 expects just obs here, so we drop the info.
        """
        out = self.env.reset(**kwargs)
        return out[0] if isinstance(out, tuple) else out

    def step(self, action):
        """
        Repeat action for 'action_repeat' steps.
        Convert Gymnasium's (obs, r, terminated, truncated, info) into
        TD-MPC2's (obs, r_sum, done, info).
        """
        total_reward = 0.0
        obs = None
        done = False
        info = {}

        for _ in range(self.action_repeat):
            step_out = self.env.step(action)

            # Gymnasium 5-tuple
            if isinstance(step_out, tuple) and len(step_out) == 5:
                obs, r, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
                info["terminated"] = bool(terminated)
            else:
                # Legacy Gym 4-tuple
                obs, r, done, info = step_out
                info["terminated"] = bool(done)

            total_reward += float(r)
            if done:
                break

        return obs, total_reward, done, info

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def render(self, *args, **kwargs):
        # Many ManiSkill2 envs accept mode='cameras' for multi-camera renders
        return self.env.render(mode="cameras")


def make_env(cfg):
    """
    Construct a ManiSkill2 environment compatible with TD-MPC2.

    Notes:
    - We use state observations here (cfg.obs must be 'state').
    - Episode length is enforced via Timeout + TimeLimit.
    """
    if cfg.task not in MANISKILL_TASKS:
        raise ValueError("Unknown task:", cfg.task)
    assert cfg.obs == "state", "This ManiSkill task wrapper currently supports only state observations."

    task_cfg = MANISKILL_TASKS[cfg.task]

    # You can tweak render_camera_cfgs if you later want image obs/videos.
    env = gym.make(
        task_cfg["env"],
        obs_mode="state",
        control_mode=task_cfg["control_mode"],
        render_camera_cfgs=dict(width=384, height=384),
    )

    # Wrap: action repeat + API bridge
    env = ManiSkillWrapper(env, cfg, action_repeat=2)

    # Enforce max episode steps via wrappers (do NOT set read-only attrs)
    max_steps = 100
    env = Timeout(env, max_episode_steps=max_steps)   # project-local wrapper
    env = TimeLimit(env, max_episode_steps=max_steps) # Gymnasium/Gym wrapper

    return env
