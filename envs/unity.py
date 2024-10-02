import gym
import numpy as np
import UnityEngine  # UnityのPython APIまたはサードパーティのライブラリ

class UnityCustomEnv:
    metadata = {}

    def __init__(self, env_path, action_repeat=1, size=(64, 64), camera=None, seed=0):
        self.env_path = env_path  # Unity環境のパス
        self.action_repeat = action_repeat
        self.size = size
        self.camera = camera if camera is not None else 0  # デフォルトのカメラID
        self.seed = seed

        # Unity環境の初期化
        self._env = self._initialize_unity_env()

        # 報酬範囲の初期化
        self.reward_range = [-np.inf, np.inf]

    def _initialize_unity_env(self):
        # Unity環境を起動し、必要な初期設定を行う
        unity_env = UnityEngine.load_env(self.env_path, seed=self.seed)  # 環境の読み込み
        return unity_env

    @property
    def observation_space(self):
        # 観測スペースの定義
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, self.size + (3,), dtype=np.uint8),
            "other_observation": gym.spaces.Box(-np.inf, np.inf, (n,), dtype=np.float32)  # 他の観測
        })

    @property
    def action_space(self):
        # アクションスペースの定義
        return gym.spaces.Box(-1.0, 1.0, (m,), dtype=np.float32)  # mはアクションの次元

    def step(self, action):
        # アクションを環境に送信し、次の状態を取得
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self.action_repeat):
            state, r, done = self._env.step(action)  # Unity環境にアクションを送信
            reward += r
            if done:
                break

        obs = self._get_observation(state)
        return obs, reward, done, {}

    def reset(self):
        # 環境をリセットして初期状態を取得
        state = self._env.reset()
        return self._get_observation(state)

    def _get_observation(self, state):
        # 状態から観測を生成するヘルパー関数
        obs = {
            "image": self.render(),
            "other_observation": state.other_observation  # 状態からの他の観測
        }
        return obs

    def render(self, *args, **kwargs):
        # 環境のレンダリング
        return self._env.render(camera_id=self.camera)  # Unity環境のレンダリングメソッドを呼び出す
