import csv
import json
import random
from pathlib import Path

import numpy as np
import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.policy import Policy, PolicySpec
from ray.rllib.utils.framework import try_import_torch

from einstein_env import EinsteinEnv

torch, nn = try_import_torch()

TRAIN_POLICY_ID = "main_policy"
RANDOM_POLICY_ID = "random_policy"
HEURISTIC_POLICY_ID = "heuristic_policy"

HISTORY_POOL_SIZE = 3
HISTORY_POLICY_IDS = [f"hist_policy_{i}" for i in range(HISTORY_POOL_SIZE)]
MAIN_POLICY_SECOND_BUCKETS = 5

ENV_CONFIG = {
    "max_steps": 200,
    "illegal_move_loss": True,
    "random_setup": False,
    "learn_setup": True,
    "win_reward": 1.0,
    "loss_reward": -1.0,
    "capture_reward": 0.03,
    "self_capture_penalty": -0.2,
    "progress_reward_scale": 0.08,
    "step_penalty": -0.01,
    "illegal_move_penalty": -1.0,
}

print("========================================")
print(f"检测到 GPU 数量: {torch.cuda.device_count()}")
print(f"PyTorch CUDA 是否可用: {torch.cuda.is_available()}")
print("========================================")


class EinsteinActionMaskModel(TorchModelV2, nn.Module):
    """
    Masked MLP model for EinStein Wuerfelt Nicht!.

    The board is encoded as discrete channels instead of raw signed numbers:
    empty + own pieces 1..6 + opponent pieces 1..6.
    Dice and candidates are also encoded as one-hot features.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        board_dim = 25 * 13
        dice_dim = 7
        candidates_dim = 2 * 7
        setup_piece_dim = 7
        phase_dim = 2
        input_dim = board_dim + dice_dim + candidates_dim + setup_piece_dim + phase_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 128)

        self.policy_head = nn.Linear(128, action_space.n)
        self.value_head = nn.Linear(128, 1)

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]

        board = obs["board"].long()
        batch_size = board.shape[0]
        flat_board = board.reshape(batch_size, -1)

        board_channels = [(flat_board == 0).float()]
        for piece_num in range(1, 7):
            board_channels.append((flat_board == piece_num).float())
        for piece_num in range(1, 7):
            board_channels.append((flat_board == -piece_num).float())
        board_features = torch.cat(board_channels, dim=1)

        dice_idx = obs["dice"].long().squeeze(1).clamp(0, 6)
        dice_features = torch.nn.functional.one_hot(dice_idx, num_classes=7).float()

        candidates = obs["candidates"].long().clamp(0, 6)
        candidate_features = (
            torch.nn.functional.one_hot(candidates, num_classes=7)
            .float()
            .reshape(batch_size, -1)
        )

        setup_piece_idx = obs["setup_piece"].long().squeeze(1).clamp(0, 6)
        setup_piece_features = torch.nn.functional.one_hot(
            setup_piece_idx, num_classes=7
        ).float()

        phase_idx = obs["phase"].long().squeeze(1).clamp(0, 1)
        phase_features = torch.nn.functional.one_hot(phase_idx, num_classes=2).float()

        action_mask = obs["action_mask"].float()

        x = torch.cat(
            [
                board_features,
                dice_features,
                candidate_features,
                setup_piece_features,
                phase_features,
            ],
            dim=1,
        )
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        logits = self.policy_head(x)
        self._value_out = self.value_head(x).squeeze(1)

        mask_sum = action_mask.sum(dim=1, keepdim=True)
        safe_mask = torch.where(mask_sum > 0, action_mask, torch.ones_like(action_mask))
        masked_logits = logits + (1.0 - safe_mask) * -1e9

        return masked_logits, state

    def value_function(self):
        return self._value_out


def env_creator(config):
    return EinsteinEnv(config)


def _obs_from_dict_batch(obs_batch):
    mask_arr = np.asarray(obs_batch["action_mask"])

    if mask_arr.ndim == 1:
        return [
            {
                "board": np.asarray(obs_batch["board"]),
                "dice": np.asarray(obs_batch["dice"]),
                "candidates": np.asarray(obs_batch["candidates"]),
                "action_mask": mask_arr,
                "phase": np.asarray(obs_batch["phase"]),
                "setup_piece": np.asarray(obs_batch["setup_piece"]),
            }
        ]

    out = []
    for i in range(mask_arr.shape[0]):
        out.append(
            {
                "board": np.asarray(obs_batch["board"])[i],
                "dice": np.asarray(obs_batch["dice"])[i],
                "candidates": np.asarray(obs_batch["candidates"])[i],
                "action_mask": mask_arr[i],
                "phase": np.asarray(obs_batch["phase"])[i],
                "setup_piece": np.asarray(obs_batch["setup_piece"])[i],
            }
        )
    return out


def _split_flat_obs(flat):
    arr = np.asarray(flat)
    if arr.shape[0] != 36:
        raise TypeError(f"Unexpected flattened Einstein obs shape: {arr.shape}")

    layouts = [
        # Common sorted Dict order: action_mask, board, candidates, dice, phase, setup_piece.
        (arr[:6], arr[6:31], arr[31:33], arr[33:34], arr[34:35], arr[35:36]),
        # Env insertion order: board, dice, candidates, action_mask, phase, setup_piece.
        (arr[28:34], arr[:25], arr[26:28], arr[25:26], arr[34:35], arr[35:36]),
        # Alternate old-stack order: board, candidates, dice, action_mask, phase, setup_piece.
        (arr[28:34], arr[:25], arr[25:27], arr[27:28], arr[34:35], arr[35:36]),
    ]

    for mask, board, candidates, dice, phase, setup_piece in layouts:
        if not np.all(np.isin(mask, [0, 1])):
            continue
        if not np.all((board >= -6) & (board <= 6)):
            continue
        if not np.all((candidates >= 0) & (candidates <= 6)):
            continue
        if not np.all((dice >= 0) & (dice <= 6)):
            continue
        if not np.all(np.isin(phase, [0, 1])):
            continue
        if not np.all((setup_piece >= 0) & (setup_piece <= 6)):
            continue

        return {
            "board": board.astype(np.int32).reshape(5, 5),
            "dice": dice.astype(np.int32),
            "candidates": candidates.astype(np.int32),
            "action_mask": mask.astype(np.float32),
            "phase": phase.astype(np.int32),
            "setup_piece": setup_piece.astype(np.int32),
        }

    raise TypeError(f"Unable to split flattened Einstein obs: {arr.tolist()}")


def unpack_obs_batch(obs_batch):
    if isinstance(obs_batch, dict):
        return _obs_from_dict_batch(obs_batch)

    if isinstance(obs_batch, (list, tuple)):
        return list(obs_batch)

    if isinstance(obs_batch, np.ndarray):
        if obs_batch.dtype == object:
            return list(obs_batch)

        arr = np.asarray(obs_batch)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim == 2:
            return [_split_flat_obs(arr[i]) for i in range(arr.shape[0])]

    raise TypeError(f"Unsupported obs_batch type: {type(obs_batch)}")


class RandomMaskedPolicy(Policy):
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        obs_list = unpack_obs_batch(obs_batch)
        actions = []

        for obs in obs_list:
            mask = np.asarray(obs["action_mask"])
            legal = np.flatnonzero(mask).tolist()
            actions.append(random.choice(legal) if legal else 0)

        return np.array(actions, dtype=np.int64), [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass


def heuristic_action_from_obs(obs):
    mask = np.asarray(obs["action_mask"])
    legal = np.flatnonzero(mask).tolist()
    if not legal:
        return 0

    phase = int(np.asarray(obs.get("phase", [1])).reshape(-1)[0])
    if phase == 0:
        return int(legal[0])

    board = np.asarray(obs["board"])
    candidates = np.asarray(obs["candidates"]).reshape(-1)
    directions = [(0, 1), (1, 0), (1, 1)]

    best_action = int(legal[0])
    best_score = -float("inf")

    for action in legal:
        slot, dir_idx = divmod(int(action), 3)
        if slot >= len(candidates) or dir_idx >= len(directions):
            continue

        piece_num = int(candidates[slot])
        if piece_num <= 0:
            continue

        positions = np.argwhere(board == piece_num)
        if len(positions) == 0:
            continue

        r, c = (int(x) for x in positions[0])
        dr, dc = directions[dir_idx]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < 5 and 0 <= nc < 5):
            continue

        target_cell = int(board[nr, nc])
        if target_cell > 0:
            continue

        old_dist = max(4 - r, 4 - c)
        new_dist = max(4 - nr, 4 - nc)
        progress_delta = old_dist - new_dist

        score = 10.0 * progress_delta - 0.5 * new_dist
        if target_cell < 0:
            score += 20.0 + abs(target_cell)
        if (nr, nc) == (4, 4):
            score += 100.0

        if score > best_score:
            best_score = score
            best_action = int(action)

    return best_action


class HeuristicMaskedPolicy(Policy):
    def compute_actions(
        self,
        obs_batch,
        state_batches=None,
        prev_action_batch=None,
        prev_reward_batch=None,
        info_batch=None,
        episodes=None,
        **kwargs,
    ):
        obs_list = unpack_obs_batch(obs_batch)
        actions = [heuristic_action_from_obs(obs) for obs in obs_list]
        return np.array(actions, dtype=np.int64), [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass


def get_episode_hash(episode):
    raw_episode_id = getattr(episode, "id_", None)
    if raw_episode_id is None:
        raw_episode_id = getattr(episode, "episode_id", None)
    if raw_episode_id is None:
        raw_episode_id = str(episode)
    return hash(str(raw_episode_id))


def choose_opponent_policy(episode_hash):
    bucket = episode_hash % 10

    if bucket < 3:
        return RANDOM_POLICY_ID

    if bucket < 6:
        return HEURISTIC_POLICY_ID

    hist_idx = (episode_hash // 10) % HISTORY_POOL_SIZE
    return HISTORY_POLICY_IDS[hist_idx]


def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    episode_hash = get_episode_hash(episode)
    main_as_second = episode_hash % 10 < MAIN_POLICY_SECOND_BUCKETS
    opponent_policy_id = choose_opponent_policy(episode_hash)

    if main_as_second:
        return TRAIN_POLICY_ID if agent_id == "player_2" else opponent_policy_id

    return TRAIN_POLICY_ID if agent_id == "player_1" else opponent_policy_id


def summarize_role_stats(stats):
    total = sum(stats.values())
    if total == 0:
        return {"total": 0, "win_rate": 0.0, "non_loss_rate": 0.0}

    return {
        "total": total,
        "win_rate": stats["win"] / total,
        "non_loss_rate": (stats["win"] + stats["draw"]) / total,
    }


def random_action(obs):
    legal = np.flatnonzero(obs["action_mask"]).tolist()
    return random.choice(legal) if legal else 0


def heuristic_action(obs):
    return heuristic_action_from_obs(obs)


def model_action(algo, obs):
    action = algo.compute_single_action(
        obs,
        policy_id=TRAIN_POLICY_ID,
        explore=False,
    )
    if isinstance(action, tuple):
        action = action[0]
    return int(action)


def evaluate_against_opponent(algo, opponent_action_fn, opponent_name, num_games=200):
    first_stats = {"win": 0, "draw": 0, "loss": 0}
    second_stats = {"win": 0, "draw": 0, "loss": 0}
    reason_stats = {}
    step_list = []

    for game_idx in range(num_games):
        env = EinsteinEnv(ENV_CONFIG)
        obs, _ = env.reset()

        model_player = "player_1" if game_idx % 2 == 0 else "player_2"
        role_stats = first_stats if model_player == "player_1" else second_stats

        done = False
        while not done:
            player = env.current_player
            current_obs = obs.get(player)
            if current_obs is None:
                current_obs = env._get_obs(player)

            if player == model_player:
                action = model_action(algo, current_obs)
            else:
                action = opponent_action_fn(current_obs)

            obs, rewards, terminateds, truncateds, infos = env.step({player: action})
            done = terminateds["__all__"] or truncateds["__all__"]

        step_list.append(env.step_count)
        reason = env.win_reason or "draw_or_truncated"
        reason_stats[reason] = reason_stats.get(reason, 0) + 1

        if env.winner == model_player:
            role_stats["win"] += 1
        elif env.winner is None:
            role_stats["draw"] += 1
        else:
            role_stats["loss"] += 1

    win = first_stats["win"] + second_stats["win"]
    draw = first_stats["draw"] + second_stats["draw"]
    loss = first_stats["loss"] + second_stats["loss"]
    avg_steps = sum(step_list) / len(step_list) if step_list else 0.0
    first_summary = summarize_role_stats(first_stats)
    second_summary = summarize_role_stats(second_stats)
    win_rate = win / num_games
    non_loss_rate = (win + draw) / num_games
    worst_role_win_rate = min(first_summary["win_rate"], second_summary["win_rate"])
    worst_role_non_loss_rate = min(
        first_summary["non_loss_rate"], second_summary["non_loss_rate"]
    )

    print(f"\n================ 评估结果：对{opponent_name} ================")
    print(f"总对局数: {num_games}")
    print(f"胜: {win} | 平/截断: {draw} | 负: {loss}")
    print(f"胜率: {win_rate:.3f}")
    print(f"不败率: {non_loss_rate:.3f}")
    print(f"先后手最弱胜率: {worst_role_win_rate:.3f}")
    print(f"先后手最弱不败率: {worst_role_non_loss_rate:.3f}")
    print(f"平均步数: {avg_steps:.2f}")
    print(f"结束原因: {reason_stats}")

    print("\n--- 模型先手(player_1) ---")
    if first_summary["total"] > 0:
        print(
            f"胜: {first_stats['win']} | 平/截断: {first_stats['draw']} | 负: {first_stats['loss']} "
            f"| 胜率: {first_summary['win_rate']:.3f} "
            f"| 不败率: {first_summary['non_loss_rate']:.3f}"
        )

    print("\n--- 模型后手(player_2) ---")
    if second_summary["total"] > 0:
        print(
            f"胜: {second_stats['win']} | 平/截断: {second_stats['draw']} | 负: {second_stats['loss']} "
            f"| 胜率: {second_summary['win_rate']:.3f} "
            f"| 不败率: {second_summary['non_loss_rate']:.3f}"
        )

    print("====================================================\n")
    return {
        "wins": win,
        "draws": draw,
        "losses": loss,
        "opponent_name": opponent_name,
        "avg_steps": avg_steps,
        "reason_stats": reason_stats,
        "first_stats": first_stats,
        "second_stats": second_stats,
        "first_win_rate": first_summary["win_rate"],
        "second_win_rate": second_summary["win_rate"],
        "first_non_loss_rate": first_summary["non_loss_rate"],
        "second_non_loss_rate": second_summary["non_loss_rate"],
        "worst_role_win_rate": worst_role_win_rate,
        "worst_role_non_loss_rate": worst_role_non_loss_rate,
        "win_rate": win_rate,
        "non_loss_rate": non_loss_rate,
    }


def evaluate_against_random(algo, num_games=200):
    return evaluate_against_opponent(
        algo,
        random_action,
        "随机玩家",
        num_games=num_games,
    )


def evaluate_against_heuristic(algo, num_games=200):
    return evaluate_against_opponent(
        algo,
        heuristic_action,
        "启发式玩家",
        num_games=num_games,
    )


def nested_get(d, path, default=None):
    cur = d
    for key in path.split("/"):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def training_record(result, iteration):
    env_stats = result.get("env_runners", {})
    if not env_stats:
        env_stats = result.get("sampler_results", {})
    learner = nested_get(result, "info/learner/main_policy/learner_stats", {}) or {}
    policy_rewards = env_stats.get("policy_reward_mean", {})

    return {
        "iteration": iteration,
        "time_total_s": result.get("time_total_s"),
        "num_env_steps_sampled": result.get("num_env_steps_sampled"),
        "episode_return_mean": env_stats.get("episode_return_mean"),
        "episode_reward_mean": env_stats.get("episode_reward_mean"),
        "episode_len_mean": env_stats.get("episode_len_mean"),
        "episodes_this_iter": env_stats.get("episodes_this_iter"),
        "main_policy_reward_mean": policy_rewards.get(TRAIN_POLICY_ID),
        "random_policy_reward_mean": policy_rewards.get(RANDOM_POLICY_ID),
        "heuristic_policy_reward_mean": policy_rewards.get(HEURISTIC_POLICY_ID),
        "hist_policy_0_reward_mean": policy_rewards.get("hist_policy_0"),
        "hist_policy_1_reward_mean": policy_rewards.get("hist_policy_1"),
        "hist_policy_2_reward_mean": policy_rewards.get("hist_policy_2"),
        "total_loss": learner.get("total_loss"),
        "policy_loss": learner.get("policy_loss"),
        "vf_loss": learner.get("vf_loss"),
        "entropy": learner.get("entropy"),
    }


def flatten_eval_stats(stats, iteration, checkpoint_path=None):
    first = stats.get("first_stats", {})
    second = stats.get("second_stats", {})

    return {
        "iteration": iteration,
        "opponent_name": stats.get("opponent_name"),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "wins": stats.get("wins"),
        "draws": stats.get("draws"),
        "losses": stats.get("losses"),
        "win_rate": stats.get("win_rate"),
        "non_loss_rate": stats.get("non_loss_rate"),
        "avg_steps": stats.get("avg_steps"),
        "first_win": first.get("win"),
        "first_draw": first.get("draw"),
        "first_loss": first.get("loss"),
        "first_win_rate": stats.get("first_win_rate"),
        "first_non_loss_rate": stats.get("first_non_loss_rate"),
        "second_win": second.get("win"),
        "second_draw": second.get("draw"),
        "second_loss": second.get("loss"),
        "second_win_rate": stats.get("second_win_rate"),
        "second_non_loss_rate": stats.get("second_non_loss_rate"),
        "worst_role_win_rate": stats.get("worst_role_win_rate"),
        "worst_role_non_loss_rate": stats.get("worst_role_non_loss_rate"),
    }


def reason_rows(stats, iteration):
    return [
        {
            "iteration": iteration,
            "opponent_name": stats.get("opponent_name"),
            "reason": reason,
            "count": count,
        }
        for reason, count in (stats.get("reason_stats") or {}).items()
    ]


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_histories(
    train_history,
    eval_history,
    reason_history,
    training_csv,
    eval_csv,
    reason_csv,
):
    write_csv(training_csv, train_history)
    write_csv(eval_csv, eval_history)
    write_csv(reason_csv, reason_history)


if __name__ == "__main__":
    TRAIN_ITERS = 1000
    EVAL_GAMES = 1000
    EVAL_EVERY = 50
    PERIODIC_EVAL_GAMES = 500

    CHECKPOINT_ROOT = Path("./checkpoints_einstein_setup").absolute()
    OUTPUT_DIR = Path("./notebook_outputs/einstein")
    TRAINING_CSV = OUTPUT_DIR / "training_history.csv"
    EVAL_CSV = OUTPUT_DIR / "eval_history.csv"
    REASON_CSV = OUTPUT_DIR / "reason_history.csv"
    FINAL_STATS_JSON = OUTPUT_DIR / "final_stats.json"

    ray.init(ignore_reinit_error=True)

    env_name = "einstein_env"
    model_name = "einstein_action_mask_model"

    register_env(env_name, env_creator)
    ModelCatalog.register_custom_model(model_name, EinsteinActionMaskModel)

    temp_env = EinsteinEnv(ENV_CONFIG)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name, env_config=ENV_CONFIG)
        .framework("torch")
        .env_runners(
            num_env_runners=2,
            num_gpus_per_env_runner=0,
        )
        .resources(num_gpus=0)
        .training(
            gamma=0.99,
            lr=1e-4,
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=10,
            entropy_coeff=0.01,
            model={"custom_model": model_name},
        )
        .multi_agent(
            policies={
                TRAIN_POLICY_ID: PolicySpec(None, obs_space, act_space, {}),
                RANDOM_POLICY_ID: PolicySpec(
                    policy_class=RandomMaskedPolicy,
                    observation_space=obs_space,
                    action_space=act_space,
                    config={},
                ),
                HEURISTIC_POLICY_ID: PolicySpec(
                    policy_class=HeuristicMaskedPolicy,
                    observation_space=obs_space,
                    action_space=act_space,
                    config={},
                ),
                **{
                    pid: PolicySpec(None, obs_space, act_space, {})
                    for pid in HISTORY_POLICY_IDS
                },
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=[TRAIN_POLICY_ID],
        )
    )

    algo = config.build_algo()
    print("开始训练 Einstein PPO(随机 + 启发式 + 历史池版)...")

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"checkpoint 保存目录: {CHECKPOINT_ROOT}")
    print(f"训练输出目录: {OUTPUT_DIR.absolute()}")

    main_weights = algo.get_policy(TRAIN_POLICY_ID).get_weights()
    algo.set_weights({pid: main_weights for pid in HISTORY_POLICY_IDS})

    best_score = -1.0
    best_win_rate = -1.0
    best_checkpoint_path = None
    history_update_idx = 0
    train_history = []
    eval_history = []
    reason_history = []

    for i in range(TRAIN_ITERS):
        result = algo.train()
        iteration = i + 1
        train_history.append(training_record(result, iteration))

        if iteration % 10 == 0:
            rec = train_history[-1]
            print(
                f"[{iteration:04d}] "
                f"return_mean={rec['episode_return_mean']} | "
                f"len_mean={rec['episode_len_mean']} | "
                f"main_reward={rec['main_policy_reward_mean']}"
            )

        if iteration % EVAL_EVERY == 0:
            print(f"\n===== 第 {iteration} 轮后开始评估 =====")
            random_stats = evaluate_against_random(algo, num_games=PERIODIC_EVAL_GAMES)
            heuristic_stats = evaluate_against_heuristic(
                algo, num_games=PERIODIC_EVAL_GAMES
            )

            checkpoint_dir = CHECKPOINT_ROOT / f"iter_{iteration:04d}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = algo.save(str(checkpoint_dir))
            print(f"已保存 checkpoint: {checkpoint_path}")

            eval_history.append(
                flatten_eval_stats(random_stats, iteration, checkpoint_path)
            )
            eval_history.append(
                flatten_eval_stats(heuristic_stats, iteration, checkpoint_path)
            )
            reason_history.extend(reason_rows(random_stats, iteration))
            reason_history.extend(reason_rows(heuristic_stats, iteration))

            score = min(
                random_stats["worst_role_win_rate"],
                heuristic_stats["worst_role_win_rate"],
            )
            if score > best_score:
                best_score = score
                best_win_rate = min(
                    random_stats["win_rate"],
                    heuristic_stats["win_rate"],
                )
                best_checkpoint_path = checkpoint_path
                print(
                    "新的最佳模型！"
                    f"robust_worst_role_win_rate = {best_score:.3f} | "
                    f"robust_overall_win_rate = {best_win_rate:.3f}"
                )

            target_hist_pid = HISTORY_POLICY_IDS[history_update_idx % HISTORY_POOL_SIZE]
            main_weights = algo.get_policy(TRAIN_POLICY_ID).get_weights()
            algo.set_weights({target_hist_pid: main_weights})
            print(f"已将 main_policy 同步到 {target_hist_pid}")
            history_update_idx += 1

            save_histories(
                train_history,
                eval_history,
                reason_history,
                TRAINING_CSV,
                EVAL_CSV,
                REASON_CSV,
            )

    final_checkpoint_dir = CHECKPOINT_ROOT / "final"
    final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    final_checkpoint = algo.save(str(final_checkpoint_dir))
    print(f"\n最终 checkpoint 已保存: {final_checkpoint}")
    print(f"\n最佳 checkpoint: {best_checkpoint_path}")
    print(f"最佳鲁棒整体胜率: {best_win_rate:.3f}")
    print(f"最佳鲁棒先后手最弱胜率: {best_score:.3f}")

    algo.stop()

    if best_checkpoint_path is None:
        print("没有找到最佳 checkpoint，跳过最终评估。")
        save_histories(
            train_history,
            eval_history,
            reason_history,
            TRAINING_CSV,
            EVAL_CSV,
            REASON_CSV,
        )
        ray.shutdown()
    else:
        best_algo = config.build_algo()
        best_algo.restore(best_checkpoint_path)

        final_random_stats = evaluate_against_random(best_algo, num_games=EVAL_GAMES)
        final_heuristic_stats = evaluate_against_heuristic(
            best_algo, num_games=EVAL_GAMES
        )
        print(f"最佳模型最终随机胜率: {final_random_stats['win_rate']:.3f}")
        print(f"最佳模型最终启发式胜率: {final_heuristic_stats['win_rate']:.3f}")
        print(
            "最佳模型最终鲁棒先后手最弱胜率: "
            f"{min(final_random_stats['worst_role_win_rate'], final_heuristic_stats['worst_role_win_rate']):.3f}"
        )

        with open(FINAL_STATS_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "env_config": ENV_CONFIG,
                    "final_checkpoint": str(final_checkpoint),
                    "best_checkpoint": str(best_checkpoint_path),
                    "best_score": best_score,
                    "best_win_rate": best_win_rate,
                    "final_stats": {
                        "random_player": final_random_stats,
                        "heuristic_player": final_heuristic_stats,
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        save_histories(
            train_history,
            eval_history,
            reason_history,
            TRAINING_CSV,
            EVAL_CSV,
            REASON_CSV,
        )
        best_algo.stop()
        ray.shutdown()
