import random
from typing import Dict, List, Tuple, Optional

import numpy as np
from gymnasium.spaces import Box, Discrete, Dict as SpaceDict
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class EinsteinEnv(MultiAgentEnv):
    """
    EinStein Wuerfelt Nicht! environment (5x5).

    The action space stays Discrete(6) for both phases:
    - setup phase: action selects one of the 6 setup cells for the current piece.
    - play phase: action selects candidate slot and direction.
    """

    BOARD_SIZE = 5
    NUM_PIECES = 6

    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        self.possible_agents = ["player_1", "player_2"]
        self.agents = self.possible_agents[:]
        self._agent_ids = set(self.possible_agents)

        self.board_size = 5
        self.max_steps = int(self.config.get("max_steps", 200))
        self.illegal_move_loss = bool(self.config.get("illegal_move_loss", True))
        self.random_setup = bool(self.config.get("random_setup", True))
        self.learn_setup = bool(self.config.get("learn_setup", False))

        self.win_reward = float(self.config.get("win_reward", 1.0))
        self.loss_reward = float(self.config.get("loss_reward", -1.0))
        self.capture_reward = float(self.config.get("capture_reward", 0.15))
        self.self_capture_penalty = float(self.config.get("self_capture_penalty", 0.0))
        self.progress_reward_scale = float(
            self.config.get("progress_reward_scale", 0.02)
        )
        self.step_penalty = float(self.config.get("step_penalty", -0.005))
        self.illegal_move_penalty = float(self.config.get("illegal_move_penalty", -1.0))

        self._obs_space = SpaceDict(
            {
                "board": Box(low=-6, high=6, shape=(5, 5), dtype=np.int32),
                "dice": Box(low=0, high=6, shape=(1,), dtype=np.int32),
                "candidates": Box(low=0, high=6, shape=(2,), dtype=np.int32),
                "action_mask": Box(low=0, high=1, shape=(6,), dtype=np.float32),
                "phase": Box(low=0, high=1, shape=(1,), dtype=np.int32),
                "setup_piece": Box(low=0, high=6, shape=(1,), dtype=np.int32),
            }
        )
        self._act_space = Discrete(6)

        self.board = np.zeros((5, 5), dtype=np.int32)
        self.positions: Dict[str, Dict[int, Tuple[int, int]]] = {
            "player_1": {},
            "player_2": {},
        }

        self.current_player = "player_1"
        self.waiting_player = "player_2"
        self.current_dice = 0
        self.current_candidates: List[Optional[int]] = [None, None]
        self.setup_phase = False
        self.setup_piece = 0
        self.setup_step_count = 0
        self.step_count = 0
        self.done = False
        self.winner: Optional[str] = None
        self.win_reason: Optional[str] = None

    @property
    def observation_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._act_space

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.agents = self.possible_agents[:]
        self.board.fill(0)
        self.positions = {"player_1": {}, "player_2": {}}
        self.step_count = 0
        self.setup_step_count = 0
        self.done = False
        self.winner = None
        self.win_reason = None
        self.current_dice = 0
        self.current_candidates = [None, None]

        if self.learn_setup:
            self.setup_phase = True
            self.setup_piece = 1
            self.current_player = "player_1"
            self.waiting_player = "player_2"
        else:
            self.setup_phase = False
            self.setup_piece = 0
            self._setup_initial_positions()
            self._start_play_phase()

        obs = {self.current_player: self._get_obs(self.current_player)}
        infos = {self.current_player: self._get_info(self.current_player)}
        return obs, infos

    def _setup_cells(self, player: str) -> List[Tuple[int, int]]:
        if player == "player_1":
            return [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)]
        return [(4, 4), (4, 3), (4, 2), (3, 4), (3, 3), (2, 4)]

    def _setup_initial_positions(self):
        p1_cells = self._setup_cells("player_1")
        p2_cells = self._setup_cells("player_2")

        if self.random_setup:
            p1_nums = list(range(1, 7))
            p2_nums = list(range(1, 7))
            random.shuffle(p1_nums)
            random.shuffle(p2_nums)
        else:
            p1_nums = [1, 2, 3, 4, 5, 6]
            p2_nums = [1, 2, 3, 4, 5, 6]

        for num, (r, c) in zip(p1_nums, p1_cells):
            self.board[r, c] = num
            self.positions["player_1"][num] = (r, c)

        for num, (r, c) in zip(p2_nums, p2_cells):
            self.board[r, c] = -num
            self.positions["player_2"][num] = (r, c)

    def _start_play_phase(self):
        self.setup_phase = False
        self.setup_piece = 0
        self.current_player = "player_1"
        self.waiting_player = "player_2"
        self.current_dice = self._roll_dice()
        self.current_candidates = self._get_candidate_pieces(
            self.current_player, self.current_dice
        )

    def _roll_dice(self) -> int:
        return random.randint(1, 6)

    def _get_candidate_pieces(
        self, player: str, dice_value: int
    ) -> List[Optional[int]]:
        alive = self.positions[player]
        if not alive:
            return [None, None]

        if dice_value in alive:
            return [dice_value, None]

        lower = None
        higher = None

        for x in range(dice_value - 1, 0, -1):
            if x in alive:
                lower = x
                break

        for x in range(dice_value + 1, 7):
            if x in alive:
                higher = x
                break

        candidates = []
        if lower is not None:
            candidates.append(lower)
        if higher is not None:
            candidates.append(higher)

        if len(candidates) == 0:
            return [None, None]
        if len(candidates) == 1:
            return [candidates[0], None]
        return [candidates[0], candidates[1]]

    def _piece_directions(self, player: str) -> List[Tuple[int, int]]:
        if player == "player_1":
            return [(0, 1), (1, 0), (1, 1)]
        return [(0, -1), (-1, 0), (-1, -1)]

    def _cell_belongs_to_player(self, cell_value: int, player: str) -> bool:
        if cell_value == 0:
            return False
        return cell_value > 0 if player == "player_1" else cell_value < 0

    def _decode_action(self, action: int) -> Tuple[int, int]:
        return action // 3, action % 3

    def _get_setup_action_mask(self, player: str) -> np.ndarray:
        mask = np.zeros(6, dtype=np.float32)
        for idx, (r, c) in enumerate(self._setup_cells(player)):
            if self.board[r, c] == 0:
                mask[idx] = 1.0
        return mask

    def _get_piece_valid_dirs(self, player: str, piece_num: int) -> List[int]:
        if piece_num not in self.positions[player]:
            return []

        r, c = self.positions[player][piece_num]
        valid_dirs = []

        for d_idx, (dr, dc) in enumerate(self._piece_directions(player)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.board_size and 0 <= nc < self.board_size):
                continue
            if self._cell_belongs_to_player(int(self.board[nr, nc]), player):
                continue
            valid_dirs.append(d_idx)

        return valid_dirs

    def _get_action_mask(self, player: str) -> np.ndarray:
        if self.setup_phase:
            return self._get_setup_action_mask(player)

        mask = np.zeros(6, dtype=np.float32)
        for slot in [0, 1]:
            piece_num = self.current_candidates[slot]
            if piece_num is None:
                continue
            for d in self._get_piece_valid_dirs(player, piece_num):
                mask[slot * 3 + d] = 1.0
        return mask

    def _action_to_move(
        self, player: str, action: int
    ) -> Optional[Tuple[int, Tuple[int, int]]]:
        if not (0 <= action < 6):
            return None

        slot, dir_idx = self._decode_action(action)
        piece_num = self.current_candidates[slot]
        if piece_num is None or piece_num not in self.positions[player]:
            return None

        if dir_idx not in self._get_piece_valid_dirs(player, piece_num):
            return None

        r, c = self.positions[player][piece_num]
        dr, dc = self._piece_directions(player)[dir_idx]
        return piece_num, (r + dr, c + dc)

    def _goal_distance(self, player: str, pos: Tuple[int, int]) -> int:
        r, c = pos
        if player == "player_1":
            gr, gc = 4, 4
        else:
            gr, gc = 0, 0
        return max(abs(gr - r), abs(gc - c))

    def _place_setup_piece(self, player: str, action: int) -> bool:
        if self.setup_piece < 1 or self.setup_piece > 6:
            return False
        if not (0 <= action < 6):
            return False

        r, c = self._setup_cells(player)[action]
        if self.board[r, c] != 0:
            return False

        value = self.setup_piece if player == "player_1" else -self.setup_piece
        self.board[r, c] = value
        self.positions[player][self.setup_piece] = (r, c)
        return True

    def _advance_setup_turn(self):
        self.setup_step_count += 1

        if self.current_player == "player_1":
            self.current_player = "player_2"
            self.waiting_player = "player_1"
            return

        if self.setup_piece < 6:
            self.setup_piece += 1
            self.current_player = "player_1"
            self.waiting_player = "player_2"
            return

        self._start_play_phase()

    def _move_piece(
        self, player: str, piece_num: int, target: Tuple[int, int]
    ) -> Tuple[bool, bool, int]:
        old_r, old_c = self.positions[player][piece_num]
        new_r, new_c = target

        old_dist = self._goal_distance(player, (old_r, old_c))
        new_dist = self._goal_distance(player, (new_r, new_c))
        progress_delta = old_dist - new_dist

        captured_opponent = False
        target_cell = int(self.board[new_r, new_c])

        if self._cell_belongs_to_player(target_cell, player):
            raise ValueError(
                f"Illegal self-capture attempt by {player}: "
                f"piece {piece_num} -> {(new_r, new_c)}"
            )

        if target_cell != 0:
            target_piece = abs(target_cell)
            target_owner = "player_1" if target_cell > 0 else "player_2"
            self.positions[target_owner].pop(target_piece, None)
            captured_opponent = True

        self.board[old_r, old_c] = 0
        self.board[new_r, new_c] = piece_num if player == "player_1" else -piece_num
        self.positions[player][piece_num] = (new_r, new_c)

        return captured_opponent, False, progress_delta

    def _check_winner(self) -> Optional[Tuple[str, str]]:
        if self.board[4, 4] > 0:
            return "player_1", "goal"
        if self.board[0, 0] < 0:
            return "player_2", "goal"

        if len(self.positions["player_1"]) == 0:
            return "player_2", "capture_all"
        if len(self.positions["player_2"]) == 0:
            return "player_1", "capture_all"

        return None

    def _get_obs(self, player: str) -> Dict[str, np.ndarray]:
        board_obs = self.board.astype(np.int32).copy()
        if player == "player_2":
            board_obs = np.rot90(board_obs, 2)
            board_obs = -board_obs

        candidates = np.array(
            [
                0 if self.current_candidates[0] is None else self.current_candidates[0],
                0 if self.current_candidates[1] is None else self.current_candidates[1],
            ],
            dtype=np.int32,
        )

        if self.setup_phase:
            dice = np.array([0], dtype=np.int32)
            candidates = np.array([self.setup_piece, 0], dtype=np.int32)
            setup_piece = self.setup_piece
            phase = 0
        else:
            dice = np.array([self.current_dice], dtype=np.int32)
            setup_piece = 0
            phase = 1

        return {
            "board": board_obs,
            "dice": dice,
            "candidates": candidates,
            "action_mask": self._get_action_mask(player),
            "phase": np.array([phase], dtype=np.int32),
            "setup_piece": np.array([setup_piece], dtype=np.int32),
        }

    def _get_info(self, player: str) -> Dict:
        return {
            "current_player": self.current_player,
            "dice": self.current_dice,
            "candidates": self.current_candidates[:],
            "phase": "setup" if self.setup_phase else "play",
            "setup_piece": self.setup_piece,
            "winner": self.winner,
            "win_reason": self.win_reason,
            "step_count": self.step_count,
            "setup_step_count": self.setup_step_count,
        }

    def _terminal_return(self, rewards, terminated=True, truncated=False):
        obs = {
            "player_1": self._get_obs("player_1"),
            "player_2": self._get_obs("player_2"),
        }
        terminateds = {
            "player_1": terminated,
            "player_2": terminated,
            "__all__": terminated,
        }
        truncateds = {
            "player_1": truncated,
            "player_2": truncated,
            "__all__": truncated,
        }
        infos = {
            "player_1": self._get_info("player_1"),
            "player_2": self._get_info("player_2"),
        }
        return obs, rewards, terminateds, truncateds, infos

    def step(self, action_dict):
        if self.done:
            return {}, {}, {"__all__": True}, {"__all__": False}, {}

        player = self.current_player
        opponent = self.waiting_player

        if player not in action_dict:
            raise ValueError(f"Expected action for {player}, got {action_dict}")

        action = int(action_dict[player])
        rewards = {"player_1": 0.0, "player_2": 0.0}

        if self.setup_phase:
            if not self._place_setup_piece(player, action):
                self.done = True
                self.winner = opponent
                self.win_reason = "illegal_setup"
                rewards[player] = self.illegal_move_penalty
                rewards[opponent] = self.win_reward
                return self._terminal_return(rewards)

            self._advance_setup_turn()
            obs = {self.current_player: self._get_obs(self.current_player)}
            terminateds = {"player_1": False, "player_2": False, "__all__": False}
            truncateds = {"player_1": False, "player_2": False, "__all__": False}
            infos = {self.current_player: self._get_info(self.current_player)}
            return obs, rewards, terminateds, truncateds, infos

        chosen = self._action_to_move(player, action)
        if chosen is None:
            if self.illegal_move_loss:
                self.done = True
                self.winner = opponent
                self.win_reason = "illegal_move"
                rewards[player] = self.illegal_move_penalty
                rewards[opponent] = self.win_reward
                return self._terminal_return(rewards)
            rewards[player] += self.illegal_move_penalty * 0.2
        else:
            piece_num, target = chosen
            captured, self_captured, progress_delta = self._move_piece(player, piece_num, target)

            rewards[player] += self.step_penalty
            rewards[player] += self.progress_reward_scale * float(progress_delta)
            if captured:
                rewards[player] += self.capture_reward
                rewards[opponent] -= self.capture_reward
            if self_captured:
                rewards[player] += self.self_capture_penalty

        self.step_count += 1

        terminal = self._check_winner()
        if terminal is not None:
            winner, reason = terminal
            self.done = True
            self.winner = winner
            self.win_reason = reason
            loser = "player_1" if winner == "player_2" else "player_2"
            rewards[winner] += self.win_reward
            rewards[loser] += self.loss_reward
            return self._terminal_return(rewards)

        if self.step_count >= self.max_steps:
            self.done = True
            return self._terminal_return(rewards, terminated=False, truncated=True)

        self.current_player, self.waiting_player = self.waiting_player, self.current_player
        self.current_dice = self._roll_dice()
        self.current_candidates = self._get_candidate_pieces(
            self.current_player, self.current_dice
        )

        next_mask = self._get_action_mask(self.current_player)
        if next_mask.sum() == 0:
            self.done = True
            self.winner = self.waiting_player
            self.win_reason = "no_legal_action"
            rewards[self.current_player] -= 1.0
            rewards[self.waiting_player] += 1.0
            return self._terminal_return(rewards)

        obs = {self.current_player: self._get_obs(self.current_player)}
        terminateds = {"player_1": False, "player_2": False, "__all__": False}
        truncateds = {"player_1": False, "player_2": False, "__all__": False}
        infos = {self.current_player: self._get_info(self.current_player)}
        return obs, rewards, terminateds, truncateds, infos

    def render(self):
        print("=" * 42)
        print(f"phase           : {'setup' if self.setup_phase else 'play'}")
        print(f"setup_piece     : {self.setup_piece}")
        print(f"step_count      : {self.step_count}")
        print(f"current_player  : {self.current_player}")
        print(f"dice            : {self.current_dice}")
        print(f"candidates      : {self.current_candidates}")
        print(f"winner          : {self.winner}")
        print(f"win_reason      : {self.win_reason}")
        print(self.board)
        print("=" * 42)
