# 文件名: tictactoe_env.py
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class TicTacToeEnv(MultiAgentEnv):
    metadata = {"render_modes": ["human"], "name": "tictactoe_v1"}

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.possible_agents = ["player_1", "player_2"]
        self.agents = self.possible_agents[:]
        self._agent_ids = set(self.possible_agents)

        self._obs_space = spaces.Dict(
            {
                # 0=empty, 1=self, 2=opponent
                "observation": spaces.Box(low=0, high=2, shape=(9,), dtype=np.int8),
                "action_mask": spaces.Box(low=0, high=1, shape=(9,), dtype=np.int8),
            }
        )
        self._act_space = spaces.Discrete(9)

        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = "player_1"
        self.waiting_player = "player_2"
        self.done = False
        self.winner = None
        self.step_count = 0

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._act_space

    def observe(self, agent):
        return self._get_obs(agent)

    def _get_obs(self, agent):
        my_val = 1 if agent == "player_1" else 2
        opp_val = 2 if agent == "player_1" else 1

        obs = np.zeros(9, dtype=np.int8)
        obs[self.board == my_val] = 1
        obs[self.board == opp_val] = 2

        mask = np.where(self.board == 0, 1, 0).astype(np.int8)
        return {"observation": obs, "action_mask": mask}

    def _get_info(self, agent):
        return {
            "current_player": self.current_player,
            "winner": self.winner,
            "step_count": self.step_count,
        }

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.board = np.zeros(9, dtype=np.int8)
        self.current_player = "player_1"
        self.waiting_player = "player_2"
        self.done = False
        self.winner = None
        self.step_count = 0

        obs = {self.current_player: self._get_obs(self.current_player)}
        infos = {self.current_player: self._get_info(self.current_player)}
        return obs, infos

    def check_win(self, player_val):
        b = self.board
        wins = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [0, 3, 6],
            [1, 4, 7],
            [2, 5, 8],
            [0, 4, 8],
            [2, 4, 6],
        ]
        return any(all(b[i] == player_val for i in line) for line in wins)

    def step(self, action_dict):
        if self.done:
            return {}, {}, {"__all__": True}, {"__all__": False}, {}

        agent = self.current_player
        opponent = self.waiting_player

        if agent not in action_dict:
            raise ValueError(f"Expected action for {agent}, got {action_dict}")

        action = int(action_dict[agent])
        rewards = {"player_1": 0.0, "player_2": 0.0}

        if not (0 <= action < 9) or self.board[action] != 0:
            self.done = True
            self.winner = opponent
            rewards[agent] = -1.0
            rewards[opponent] = 1.0

            obs = {
                "player_1": self._get_obs("player_1"),
                "player_2": self._get_obs("player_2"),
            }
            terminateds = {"player_1": True, "player_2": True, "__all__": True}
            truncateds = {"player_1": False, "player_2": False, "__all__": False}
            infos = {
                "player_1": self._get_info("player_1"),
                "player_2": self._get_info("player_2"),
            }
            return obs, rewards, terminateds, truncateds, infos

        player_val = 1 if agent == "player_1" else 2
        self.board[action] = player_val
        self.step_count += 1

        if self.check_win(player_val):
            self.done = True
            self.winner = agent
            rewards[agent] = 1.0
            rewards[opponent] = -1.0

            obs = {
                "player_1": self._get_obs("player_1"),
                "player_2": self._get_obs("player_2"),
            }
            terminateds = {"player_1": True, "player_2": True, "__all__": True}
            truncateds = {"player_1": False, "player_2": False, "__all__": False}
            infos = {
                "player_1": self._get_info("player_1"),
                "player_2": self._get_info("player_2"),
            }
            return obs, rewards, terminateds, truncateds, infos

        if not np.any(self.board == 0):
            self.done = True
            self.winner = None

            obs = {
                "player_1": self._get_obs("player_1"),
                "player_2": self._get_obs("player_2"),
            }
            terminateds = {"player_1": True, "player_2": True, "__all__": True}
            truncateds = {"player_1": False, "player_2": False, "__all__": False}
            infos = {
                "player_1": self._get_info("player_1"),
                "player_2": self._get_info("player_2"),
            }
            return obs, rewards, terminateds, truncateds, infos

        self.current_player, self.waiting_player = self.waiting_player, self.current_player

        obs = {self.current_player: self._get_obs(self.current_player)}
        terminateds = {"player_1": False, "player_2": False, "__all__": False}
        truncateds = {"player_1": False, "player_2": False, "__all__": False}
        infos = {self.current_player: self._get_info(self.current_player)}
        return obs, rewards, terminateds, truncateds, infos

    def render(self):
        symbols = {0: ".", 1: "X", 2: "O"}
        rows = []
        for r in range(3):
            rows.append(" ".join(symbols[int(x)] for x in self.board[r * 3 : (r + 1) * 3]))
        print("\n".join(rows))
