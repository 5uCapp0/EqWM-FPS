"""
env/vizdoom_env.py
==================
ViZDoom 环境封装（headless），支持 defend_the_center 与 health_gathering。

关键设计：
- 屏幕固定 resize 到 64x64，RGB，归一化到 [0,1]，形状 (3,64,64)。
- 动作为离散索引：
    health_gathering:   [TURN_LEFT=0, TURN_RIGHT=1, MOVE_FORWARD=2]
    defend_the_center:  [TURN_LEFT=0, TURN_RIGHT=1, ATTACK=2]
- 水平翻转时 0<->1（见 utils/transforms.py）。
- headless: set_window_visible(False)，Windows 下无需虚拟显示器。
"""

import numpy as np
import vizdoom as vzd


class ViZDoomEnv:
    def __init__(self, scenario: str = "health_gathering", frame_skip: int = 4,
                 resolution: int = 64, seed: int = 0):
        """
        Args:
            scenario: 'health_gathering' 或 'defend_the_center'。
            frame_skip: 每个动作重复的 tic 数（加速）。
            resolution: 输出帧边长（正方形）。
            seed: 随机种子。
        """
        assert scenario in ("health_gathering", "defend_the_center")
        self.scenario = scenario
        self.frame_skip = frame_skip
        self.resolution = resolution

        self.game = vzd.DoomGame()
        cfg = vzd.scenarios_path + f"/{scenario}.cfg"
        self.game.load_config(cfg)
        # 固定小分辨率以加速渲染（内部仍按 cfg 渲染，再 resize）
        self.game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        self.game.set_screen_format(vzd.ScreenFormat.RGB24)
        self.game.set_window_visible(False)
        self.game.set_mode(vzd.Mode.PLAYER)
        self.game.set_seed(seed)
        self.game.init()

        # 离散动作表
        if scenario == "health_gathering":
            self.actions = [
                [True, False, False],   # 0 LEFT
                [False, True, False],   # 1 RIGHT
                [False, False, True],   # 2 FORWARD
            ]
        else:  # defend_the_center
            self.actions = [
                [True, False, False],   # 0 LEFT
                [False, True, False],   # 1 RIGHT
                [False, False, True],   # 2 ATTACK
            ]
        self.n_actions = len(self.actions)
        self._last_reward = 0.0

    def reset(self) -> np.ndarray:
        """重置 episode，返回首帧 obs (3,H,W) float32 in [0,1]。"""
        self.game.new_episode()
        return self._get_obs()

    def _get_obs(self) -> np.ndarray:
        state = self.game.get_state()
        if state is None:
            return np.zeros((3, self.resolution, self.resolution), dtype=np.float32)
        buf = state.screen_buffer  # 可能为 (H,W,3) 或 (3,H,W)
        if buf.shape[-1] == 3 and buf.shape[0] != 3:
            buf = np.transpose(buf, (2, 0, 1))  # HWC -> CHW
        obs = self._resize(buf)
        return (obs.astype(np.float32) / 255.0)

    def _resize(self, buf: np.ndarray) -> np.ndarray:
        """CHW 双线性-ish resize 到 (3, R, R)。用 numpy 均匀采样避免 opencv 依赖。"""
        _, H, W = buf.shape
        R = self.resolution
        ys = (np.linspace(0, H - 1, R)).astype(np.int64)
        xs = (np.linspace(0, W - 1, R)).astype(np.int64)
        return buf[:, ys][:, :, xs]

    def step(self, action_idx: int):
        """执行离散动作。

        Returns:
            obs (3,R,R) float32, reward float, done bool, info dict
        """
        reward = self.game.make_action(self.actions[action_idx], self.frame_skip)
        done = self.game.is_episode_finished()
        obs = self._get_obs() if not done else np.zeros(
            (3, self.resolution, self.resolution), dtype=np.float32)
        self._last_reward = reward
        return obs, float(reward), done, {}

    def close(self):
        self.game.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
