#!/usr/bin/env python
# coding: utf-8

# # Tic-Tac-Toe Reward Sweep Overnight Experiments
# 
# 这个 notebook 用来重新跑一批更适合写进论文的 Tic-Tac-Toe 奖励函数对比实验。
# 
# 实验计划：
# 
# - `draw_reward = 0.0`：只奖励胜利，平局不给正奖励；
# - `draw_reward = 0.2`：老师版本/温和平局奖励；
# - `draw_reward = 0.5`：更强的不败倾向奖励；
# - 每个 `draw_reward` 跑 500 轮 x 3 次；
# - 每个 `draw_reward` 再跑 1000 轮 x 1 次，用来观察“训练更久是否一定更好”。
# 
# 输出会保存到：
# 
# - `notebook_outputs/tictactoe_reward_sweep_overnight/`
# - `paper_figures/tictactoe_reward_sweep_overnight/`
# 
# 建议晚上运行前先重启 kernel，然后选择 `Run All`。如果中途断掉，本 notebook 默认会跳过已经完成并保存 `run_summary.json` 的 run。
# 

# In[ ]:


import importlib.util
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

# 这批 Tic-Tac-Toe 实验不需要 GPU。显式关闭 CUDA 可避免 WSL/DXG 与 Ray
# 的偶发交互问题；关闭 Ray dashboard/metrics 可减少 GCS 旁路服务。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
os.environ.setdefault("RAY_DISABLE_DASHBOARD", "1")
os.environ.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import torch

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

PROJECT_ROOT = Path.cwd()
TEACHER_DIR = PROJECT_ROOT / "teacher"


def teacher_file(*names):
    for name in names:
        path = TEACHER_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError("Missing teacher file. Tried: " + ", ".join(names))


TEACHER_FILES = {
    "env": teacher_file("tictactoe_env.py", "tictactoe_env(1).py"),
    "minimax": teacher_file("minimax_utils.py", "minimax_utils(1).py"),
    "train": teacher_file("train_tictactoe.py", "train_tictactoe(1).py"),
}


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# 先用标准模块名加载 env/minimax，保证 teacher/train_tictactoe.py import 到 teacher 目录中的版本。
env_teacher = load_module("tictactoe_env", TEACHER_FILES["env"])
minimax_teacher = load_module("minimax_utils", TEACHER_FILES["minimax"])
ttt = load_module("teacher_train_tictactoe_reward_sweep", TEACHER_FILES["train"])

TicTacToeEnv = env_teacher.TicTacToeEnv

print("Loaded teacher files:")
for key, path in TEACHER_FILES.items():
    print(f"  {key}: {path}")
print("Torch CUDA available:", torch.cuda.is_available())


# In[ ]:


# ===== Overnight reward sweep plan =====
DRAW_REWARDS = [0.0, 0.2, 0.5]
SEEDS_500 = [101, 202, 303]
SEED_1000 = 101  # 与 500 轮第 1 次相同，便于比较“训练更久”的影响

EVAL_EVERY = 50
PERIODIC_EVAL_GAMES = 300      # 周期评估用较小局数，节约整夜运行时间
FINAL_EVAL_GAMES = 1000        # 最终评估仍然使用 1000 局，适合写进论文
PRINT_EVERY = 10
HISTORY_UPDATE_EVERY = 50

NUM_GPUS = 0
NUM_ENV_RUNNERS = 0             # notebook 动态加载 teacher 模块时，本地 runner 最稳定
RAY_LOCAL_MODE = True            # 避免长时间实验中 GCS/raylet 远程调度链路不稳定

SAVE_BEST_CHECKPOINTS = False    # 本批实验只需要 CSV/图；关闭 checkpoint 更稳、更省磁盘
SAVE_FINAL_CHECKPOINTS = False
SKIP_COMPLETED = True           # 断点续跑：已有 run_summary.json 的实验会跳过
COLLECT_LOSING_EXAMPLES = False # 如需额外棋谱诊断，可改为 True

MAIN_POLICY_SECOND_BUCKETS = 8
SECOND_MINIMAX_BUCKETS = 18  # out of 20, conditional on main as second
FIRST_MINIMAX_BUCKETS = 16   # out of 20, conditional on main as first

SWEEP_OUTPUT_DIR = Path("notebook_outputs/tictactoe_reward_sweep_overnight")
SWEEP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAY_TEMP_DIR = Path("/tmp/ttt_ray").absolute()
RAY_TEMP_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_ROOT = Path("checkpoints_tictactoe_reward_sweep_overnight").absolute()
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

FIGURE_DIR = Path("paper_figures/tictactoe_reward_sweep_overnight")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def draw_reward_tag(draw_reward):
    return str(draw_reward).replace(".", "p")


plan_rows = []
for draw_reward in DRAW_REWARDS:
    for repeat, seed in enumerate(SEEDS_500, start=1):
        plan_rows.append({
            "run_id": f"draw_{draw_reward_tag(draw_reward)}_iters_500_seed_{seed}",
            "draw_reward": draw_reward,
            "train_iters": 500,
            "repeat": repeat,
            "seed": seed,
            "group": "500x3",
        })
    plan_rows.append({
        "run_id": f"draw_{draw_reward_tag(draw_reward)}_iters_1000_seed_{SEED_1000}",
        "draw_reward": draw_reward,
        "train_iters": 1000,
        "repeat": 1,
        "seed": SEED_1000,
        "group": "1000x1",
    })

EXPERIMENT_PLAN = pd.DataFrame(plan_rows)
EXPERIMENT_PLAN.to_csv(SWEEP_OUTPUT_DIR / "experiment_plan.csv", index=False)

print("Experiment plan:")
try:
    display(EXPERIMENT_PLAN)
except NameError:
    print(EXPERIMENT_PLAN)
print("Total runs:", len(EXPERIMENT_PLAN))


# ## 运行规模说明
# 
# 总共会跑 12 个 run：
# 
# - 500 轮：`3 个 draw_reward x 3 个 seed = 9` 个 run；
# - 1000 轮：`3 个 draw_reward x 1 个 seed = 3` 个 run。
# 
# 周期评估每 50 轮进行一次，每次对 random 和 minimax 各评估 300 局；最终评估对 random 和 minimax 各评估 1000 局。这样既能保留曲线，又不会让整夜运行时间过分膨胀。
# 

# In[ ]:


# 验证 teacher 模型是否仍然使用 One-Hot 通道分离输入
import inspect

model_source = inspect.getsource(ttt.TicTacToeMaskModel)
print("Model class loaded from:", inspect.getsourcefile(ttt.TicTacToeMaskModel))
for line in model_source.splitlines():
    if any(key in line for key in [
        "nn.Linear(18",
        "my_pieces",
        "opp_pieces",
        "encoded_board",
        "fc1(encoded_board)",
    ]):
        print(line)

assert "nn.Linear(18, 128)" in model_source
assert "my_pieces = (board == 1.0).float()" in model_source
assert "opp_pieces = (board == 2.0).float()" in model_source
assert "encoded_board = torch.cat([my_pieces, opp_pieces], dim=-1)" in model_source
print("Verified: model uses a channel-separated 18-dimensional board input.")


# In[ ]:


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def notebook_choose_opponent_policy(episode_hash, main_as_second=False):
    """Teacher-aligned profile: more minimax games, especially as second player."""
    opponent_bucket = (episode_hash // 10) % 20

    if main_as_second:
        if opponent_bucket < SECOND_MINIMAX_BUCKETS:
            return ttt.MINIMAX_POLICY_ID
        if opponent_bucket < 19:
            return ttt.RANDOM_POLICY_ID
        hist_idx = (episode_hash // 200) % ttt.HISTORY_POOL_SIZE
        return ttt.HISTORY_POLICY_IDS[hist_idx]

    if opponent_bucket < FIRST_MINIMAX_BUCKETS:
        return ttt.MINIMAX_POLICY_ID
    if opponent_bucket < 18:
        return ttt.RANDOM_POLICY_ID
    hist_idx = (episode_hash // 200) % ttt.HISTORY_POOL_SIZE
    return ttt.HISTORY_POLICY_IDS[hist_idx]


def notebook_policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    episode_hash = ttt.get_episode_hash(episode)
    main_as_second = episode_hash % 10 < MAIN_POLICY_SECOND_BUCKETS
    opponent_policy_id = notebook_choose_opponent_policy(episode_hash, main_as_second)

    if main_as_second:
        return ttt.TRAIN_POLICY_ID if agent_id == "player_2" else opponent_policy_id
    return ttt.TRAIN_POLICY_ID if agent_id == "player_1" else opponent_policy_id


def build_tictactoe_config(env_config, seed, run_id):
    env_name = f"TicTacToe-v1-reward-sweep-{run_id}"
    model_name = f"tictactoe_reward_sweep_mask_model_{run_id}"

    def env_creator(config):
        return TicTacToeEnv(config)

    register_env(env_name, env_creator)
    ModelCatalog.register_custom_model(model_name, ttt.TicTacToeMaskModel)

    temp_env = TicTacToeEnv(env_config)
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env_name, env_config=env_config)
        .framework("torch")
        .resources(num_gpus=NUM_GPUS)
        .env_runners(
            num_env_runners=NUM_ENV_RUNNERS,
            num_gpus_per_env_runner=0,
        )
        .training(
            model={"custom_model": model_name},
            entropy_coeff_schedule=[
                [0, 0.001],
                [1_000_000, 0.0002],
                [2_000_000, 0.0],
            ],
            lr=1e-4,
            gamma=0.99,
            train_batch_size=4000,
            minibatch_size=256,
            num_epochs=10,
        )
        .multi_agent(
            policies={
                ttt.TRAIN_POLICY_ID: PolicySpec(None, obs_space, act_space, {}),
                ttt.RANDOM_POLICY_ID: PolicySpec(
                    policy_class=ttt.RandomMaskedPolicy,
                    observation_space=obs_space,
                    action_space=act_space,
                    config={},
                ),
                ttt.MINIMAX_POLICY_ID: PolicySpec(
                    policy_class=ttt.MinimaxPolicy,
                    observation_space=obs_space,
                    action_space=act_space,
                    config={},
                ),
                **{
                    pid: PolicySpec(None, obs_space, act_space, {})
                    for pid in ttt.HISTORY_POLICY_IDS
                },
            },
            policy_mapping_fn=notebook_policy_mapping_fn,
            policies_to_train=[ttt.TRAIN_POLICY_ID],
        )
    )

    try:
        config = config.debugging(seed=seed)
    except Exception as exc:
        print("Config seed not set through RLlib debugging API:", repr(exc))

    return config


def nested_get(d, path, default=None):
    cur = d
    for key in path.split("/"):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def training_record(result, iteration):
    env_stats = result.get("env_runners") or result.get("sampler_results") or {}
    learner = nested_get(result, "info/learner/main_policy/learner_stats", {}) or {}
    return {
        "iteration": iteration,
        "time_total_s": result.get("time_total_s"),
        "num_env_steps_sampled": result.get("num_env_steps_sampled"),
        "episode_return_mean": env_stats.get("episode_return_mean"),
        "episode_reward_mean": env_stats.get("episode_reward_mean"),
        "episode_len_mean": env_stats.get("episode_len_mean"),
        "episodes_this_iter": env_stats.get("episodes_this_iter"),
        "main_policy_reward_mean": nested_get(result, "env_runners/policy_reward_mean/main_policy"),
        "random_policy_reward_mean": nested_get(result, "env_runners/policy_reward_mean/random_policy"),
        "minimax_policy_reward_mean": nested_get(result, "env_runners/policy_reward_mean/minimax_policy"),
        "total_loss": learner.get("total_loss"),
        "policy_loss": learner.get("policy_loss"),
        "vf_loss": learner.get("vf_loss"),
        "entropy": learner.get("entropy"),
    }


def checkpoint_to_path(save_result):
    checkpoint = getattr(save_result, "checkpoint", save_result)
    return str(getattr(checkpoint, "path", checkpoint))


# In[ ]:


def compute_policy_action(algo, current_obs, policy_id=ttt.TRAIN_POLICY_ID):
    action = algo.compute_single_action(
        current_obs,
        policy_id=policy_id,
        explore=False,
    )
    if isinstance(action, tuple):
        action = action[0]
    return int(action)


def summarize_role_stats(stats):
    total = sum(stats.values())
    if total == 0:
        return {"total": 0, "win_rate": 0.0, "non_loss_rate": 0.0, "loss_rate": 0.0}
    return {
        "total": total,
        "win_rate": stats["win"] / total,
        "non_loss_rate": (stats["win"] + stats["draw"]) / total,
        "loss_rate": stats["loss"] / total,
    }


def evaluate_against_opponent(algo, env_config, opponent_name, opponent_action_fn, num_games=200):
    first_stats = {"win": 0, "draw": 0, "loss": 0}
    second_stats = {"win": 0, "draw": 0, "loss": 0}
    step_list = []

    for game_idx in range(num_games):
        env = TicTacToeEnv(env_config)
        obs, _ = env.reset()

        model_player = "player_1" if game_idx % 2 == 0 else "player_2"
        role_stats = first_stats if model_player == "player_1" else second_stats

        done = False
        while not done:
            agent = env.current_player
            current_obs = obs.get(agent)
            if current_obs is None:
                current_obs = env._get_obs(agent)

            if agent == model_player:
                action = compute_policy_action(algo, current_obs)
            else:
                action = opponent_action_fn(current_obs)

            obs, rewards, terminateds, truncateds, infos = env.step({agent: action})
            done = terminateds["__all__"] or truncateds["__all__"]

        step_list.append(env.step_count)

        if env.winner == model_player:
            role_stats["win"] += 1
        elif env.winner is None:
            role_stats["draw"] += 1
        else:
            role_stats["loss"] += 1

    win = first_stats["win"] + second_stats["win"]
    draw = first_stats["draw"] + second_stats["draw"]
    loss = first_stats["loss"] + second_stats["loss"]
    first_summary = summarize_role_stats(first_stats)
    second_summary = summarize_role_stats(second_stats)
    non_loss_rate = (win + draw) / num_games
    worst_role_non_loss_rate = min(first_summary["non_loss_rate"], second_summary["non_loss_rate"])
    worst_role_loss_rate = max(first_summary["loss_rate"], second_summary["loss_rate"])

    print(f"\n================ Evaluation vs {opponent_name} ================")
    print(f"Total games: {num_games}")
    print(f"Wins: {win} | Draws: {draw} | Losses: {loss}")
    print(f"Win rate: {win / num_games:.3f}")
    print(f"Non-loss rate: {non_loss_rate:.3f}")
    print(f"Worst-role non-loss rate: {worst_role_non_loss_rate:.3f}")
    print(f"Average steps: {sum(step_list) / len(step_list):.2f}")

    return {
        "opponent_name": opponent_name,
        "wins": win,
        "draws": draw,
        "losses": loss,
        "avg_steps": sum(step_list) / len(step_list) if step_list else 0.0,
        "first_stats": first_stats,
        "second_stats": second_stats,
        "first_non_loss_rate": first_summary["non_loss_rate"],
        "second_non_loss_rate": second_summary["non_loss_rate"],
        "first_loss_rate": first_summary["loss_rate"],
        "second_loss_rate": second_summary["loss_rate"],
        "worst_role_non_loss_rate": worst_role_non_loss_rate,
        "worst_role_loss_rate": worst_role_loss_rate,
        "non_loss_rate": non_loss_rate,
        "loss_rate": loss / num_games,
        "win_rate": win / num_games,
    }


def evaluate_against_random(algo, env_config, num_games=200):
    return evaluate_against_opponent(
        algo,
        env_config=env_config,
        opponent_name="random player",
        opponent_action_fn=ttt.random_action_from_obs,
        num_games=num_games,
    )


def evaluate_against_minimax(algo, env_config, num_games=200):
    return evaluate_against_opponent(
        algo,
        env_config=env_config,
        opponent_name="minimax",
        opponent_action_fn=ttt.minimax_action_from_obs,
        num_games=num_games,
    )


def flatten_eval_stats(stats, iteration, eval_phase, checkpoint_path=None):
    first = stats.get("first_stats", {})
    second = stats.get("second_stats", {})
    return {
        "iteration": iteration,
        "eval_phase": eval_phase,
        "opponent_name": stats.get("opponent_name"),
        "checkpoint_path": checkpoint_path,
        "wins": stats.get("wins"),
        "draws": stats.get("draws"),
        "losses": stats.get("losses"),
        "win_rate": stats.get("win_rate"),
        "non_loss_rate": stats.get("non_loss_rate"),
        "loss_rate": stats.get("loss_rate"),
        "avg_steps": stats.get("avg_steps"),
        "first_win": first.get("win"),
        "first_draw": first.get("draw"),
        "first_loss": first.get("loss"),
        "first_non_loss_rate": stats.get("first_non_loss_rate"),
        "first_loss_rate": stats.get("first_loss_rate"),
        "second_win": second.get("win"),
        "second_draw": second.get("draw"),
        "second_loss": second.get("loss"),
        "second_non_loss_rate": stats.get("second_non_loss_rate"),
        "second_loss_rate": stats.get("second_loss_rate"),
        "worst_role_non_loss_rate": stats.get("worst_role_non_loss_rate"),
        "worst_role_loss_rate": stats.get("worst_role_loss_rate"),
    }


# In[ ]:


def collect_losing_games(algo, env_config, opponent_name, opponent_action_fn, max_games=500, max_records=20):
    records = []

    for game_idx in range(max_games):
        if len(records) >= max_records:
            break

        env = TicTacToeEnv(env_config)
        obs, _ = env.reset()
        model_player = "player_1" if game_idx % 2 == 0 else "player_2"
        trajectory = []

        done = False
        while not done:
            agent = env.current_player
            current_obs = obs.get(agent)
            if current_obs is None:
                current_obs = env._get_obs(agent)

            board_before = env.board.astype(int).tolist()
            legal_actions = np.flatnonzero(current_obs["action_mask"]).astype(int).tolist()

            if agent == model_player:
                action = compute_policy_action(algo, current_obs)
                actor = "model"
            else:
                action = opponent_action_fn(current_obs)
                actor = opponent_name

            obs, rewards, terminateds, truncateds, infos = env.step({agent: action})
            trajectory.append({
                "step": len(trajectory) + 1,
                "agent": agent,
                "actor": actor,
                "action": int(action),
                "legal_actions": legal_actions,
                "board_before": board_before,
                "board_after": env.board.astype(int).tolist(),
                "winner_after": env.winner,
            })
            done = terminateds["__all__"] or truncateds["__all__"]

        if env.winner is not None and env.winner != model_player:
            records.append({
                "game_idx": game_idx,
                "opponent_name": opponent_name,
                "model_player": model_player,
                "winner": env.winner,
                "steps": env.step_count,
                "trajectory": trajectory,
            })

    return records


def save_run_tables(run_dir, train_history, eval_history):
    train_df = pd.DataFrame(train_history)
    eval_df = pd.DataFrame(eval_history)
    train_df.to_csv(run_dir / "training_history.csv", index=False)
    eval_df.to_csv(run_dir / "eval_history.csv", index=False)
    if not eval_df.empty:
        eval_df[eval_df["opponent_name"] == "random player"].to_csv(run_dir / "eval_random_history.csv", index=False)
        eval_df[eval_df["opponent_name"] == "minimax"].to_csv(run_dir / "eval_minimax_history.csv", index=False)


def run_single_experiment(plan_row):
    plan_row = dict(plan_row)
    run_id = plan_row["run_id"]
    draw_reward = float(plan_row["draw_reward"])
    train_iters = int(plan_row["train_iters"])
    seed = int(plan_row["seed"])

    run_dir = SWEEP_OUTPUT_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_summary_path = run_dir / "run_summary.json"

    if SKIP_COMPLETED and run_summary_path.exists():
        print(f"\n===== Skipping completed run: {run_id} =====")
        return json.loads(run_summary_path.read_text())

    print("\n" + "=" * 80)
    print(f"Starting run: {run_id}")
    print(f"draw_reward={draw_reward} | train_iters={train_iters} | seed={seed}")
    print("=" * 80)

    set_global_seed(seed)
    env_config = {"draw_reward": draw_reward}
    checkpoint_dir = CHECKPOINT_ROOT / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_history = []
    eval_history = []
    best_score = -1.0
    best_non_loss = -1.0
    best_random_score = -1.0
    best_checkpoint_path = None
    final_checkpoint_path = None
    history_update_idx = 0
    run_start = time.time()

    ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        local_mode=RAY_LOCAL_MODE,
        _temp_dir=str(RAY_TEMP_DIR),
    )

    algo = None
    try:
        config = build_tictactoe_config(env_config=env_config, seed=seed, run_id=run_id)
        algo = config.build_algo()

        main_weights = algo.get_policy(ttt.TRAIN_POLICY_ID).get_weights()
        algo.set_weights({pid: main_weights for pid in ttt.HISTORY_POLICY_IDS})

        for i in range(train_iters):
            result = algo.train()
            iteration = i + 1
            rec = training_record(result, iteration)
            rec.update(plan_row)
            train_history.append(rec)

            if iteration % PRINT_EVERY == 0:
                print(
                    f"[{run_id} | {iteration:04d}/{train_iters}] "
                    f"return_mean={rec['episode_return_mean']} | "
                    f"len_mean={rec['episode_len_mean']} | "
                    f"main_reward={rec['main_policy_reward_mean']}"
                )

            if iteration % EVAL_EVERY == 0:
                print(f"\n===== Eval at iter {iteration} for {run_id} =====")
                random_stats = evaluate_against_random(algo, env_config, num_games=PERIODIC_EVAL_GAMES)
                minimax_stats = evaluate_against_minimax(algo, env_config, num_games=PERIODIC_EVAL_GAMES)

                checkpoint_path = None
                score = minimax_stats["worst_role_non_loss_rate"]
                non_loss = minimax_stats["non_loss_rate"]
                random_score = random_stats["worst_role_non_loss_rate"]
                improved = score > best_score or (score == best_score and random_score > best_random_score)

                if improved:
                    best_score = score
                    best_non_loss = non_loss
                    best_random_score = random_score
                    if SAVE_BEST_CHECKPOINTS:
                        checkpoint_path = checkpoint_to_path(algo.save(str(checkpoint_dir)))
                        best_checkpoint_path = checkpoint_path
                    print(
                        "New best model found: "
                        f"minimax_worst_role_non_loss={best_score:.3f} | "
                        f"minimax_non_loss={best_non_loss:.3f} | "
                        f"random_worst_role_non_loss={best_random_score:.3f}"
                    )

                for stats in (random_stats, minimax_stats):
                    row = flatten_eval_stats(stats, iteration, eval_phase="periodic", checkpoint_path=checkpoint_path)
                    row.update(plan_row)
                    eval_history.append(row)

                save_run_tables(run_dir, train_history, eval_history)

            if iteration % HISTORY_UPDATE_EVERY == 0:
                target_hist_pid = ttt.HISTORY_POLICY_IDS[history_update_idx % ttt.HISTORY_POOL_SIZE]
                main_weights = algo.get_policy(ttt.TRAIN_POLICY_ID).get_weights()
                algo.set_weights({target_hist_pid: main_weights})
                print(f"Synchronized main_policy to {target_hist_pid}")
                history_update_idx += 1

        if SAVE_FINAL_CHECKPOINTS:
            final_checkpoint_path = checkpoint_to_path(algo.save(str(checkpoint_dir)))
            print("Final checkpoint saved:", final_checkpoint_path)

        print(f"\n===== Final evaluation for {run_id} =====")
        final_random_stats = evaluate_against_random(algo, env_config, num_games=FINAL_EVAL_GAMES)
        final_minimax_stats = evaluate_against_minimax(algo, env_config, num_games=FINAL_EVAL_GAMES)

        for stats in (final_random_stats, final_minimax_stats):
            row = flatten_eval_stats(stats, train_iters, eval_phase="final", checkpoint_path=final_checkpoint_path)
            row.update(plan_row)
            eval_history.append(row)

        if COLLECT_LOSING_EXAMPLES:
            losing_examples = collect_losing_games(
                algo,
                env_config,
                opponent_name="minimax",
                opponent_action_fn=ttt.minimax_action_from_obs,
                max_games=500,
                max_records=20,
            )
            (run_dir / "losing_game_examples.json").write_text(
                json.dumps(losing_examples, indent=2, ensure_ascii=False)
            )

        elapsed_s = time.time() - run_start
        summary = {
            **plan_row,
            "status": "completed",
            "elapsed_s": elapsed_s,
            "best_checkpoint_path": best_checkpoint_path,
            "final_checkpoint_path": final_checkpoint_path,
            "best_periodic_minimax_non_loss_rate": best_non_loss,
            "best_periodic_minimax_worst_role_non_loss_rate": best_score,
            "best_periodic_random_worst_role_non_loss_rate": best_random_score,
            "final_random_wins": final_random_stats["wins"],
            "final_random_draws": final_random_stats["draws"],
            "final_random_losses": final_random_stats["losses"],
            "final_random_win_rate": final_random_stats["win_rate"],
            "final_random_non_loss_rate": final_random_stats["non_loss_rate"],
            "final_random_first_non_loss_rate": final_random_stats["first_non_loss_rate"],
            "final_random_second_non_loss_rate": final_random_stats["second_non_loss_rate"],
            "final_minimax_wins": final_minimax_stats["wins"],
            "final_minimax_draws": final_minimax_stats["draws"],
            "final_minimax_losses": final_minimax_stats["losses"],
            "final_minimax_win_rate": final_minimax_stats["win_rate"],
            "final_minimax_non_loss_rate": final_minimax_stats["non_loss_rate"],
            "final_minimax_first_non_loss_rate": final_minimax_stats["first_non_loss_rate"],
            "final_minimax_second_non_loss_rate": final_minimax_stats["second_non_loss_rate"],
            "final_minimax_worst_role_non_loss_rate": final_minimax_stats["worst_role_non_loss_rate"],
        }

        save_run_tables(run_dir, train_history, eval_history)
        run_summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"Run completed: {run_id} | elapsed={elapsed_s/60:.1f} min")
        return summary

    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()
        save_run_tables(run_dir, train_history, eval_history)


# In[ ]:


def run_reward_sweep(plan_df=EXPERIMENT_PLAN):
    all_summaries = []
    failures = []

    for _, row in plan_df.iterrows():
        try:
            summary = run_single_experiment(row)
            all_summaries.append(summary)
            pd.DataFrame(all_summaries).to_csv(SWEEP_OUTPUT_DIR / "all_runs_summary.csv", index=False)
        except Exception as exc:
            run_id = row.get("run_id", "unknown")
            print("\n" + "!" * 80)
            print(f"Run failed: {run_id}")
            traceback.print_exc()
            print("!" * 80)
            failures.append({
                "run_id": run_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            })
            (SWEEP_OUTPUT_DIR / "failed_runs.json").write_text(
                json.dumps(failures, indent=2, ensure_ascii=False)
            )
            pd.DataFrame(failures).to_csv(SWEEP_OUTPUT_DIR / "failed_runs.csv", index=False)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(SWEEP_OUTPUT_DIR / "all_runs_summary.csv", index=False)

    eval_parts = []
    train_parts = []
    for run_id in EXPERIMENT_PLAN["run_id"]:
        run_dir = SWEEP_OUTPUT_DIR / run_id
        eval_path = run_dir / "eval_history.csv"
        train_path = run_dir / "training_history.csv"
        if eval_path.exists():
            eval_parts.append(pd.read_csv(eval_path))
        if train_path.exists():
            train_parts.append(pd.read_csv(train_path))

    if eval_parts:
        pd.concat(eval_parts, ignore_index=True).to_csv(SWEEP_OUTPUT_DIR / "all_eval_history.csv", index=False)
    if train_parts:
        pd.concat(train_parts, ignore_index=True).to_csv(SWEEP_OUTPUT_DIR / "all_training_history.csv", index=False)

    print("\nReward sweep finished.")
    print("Summary saved to:", SWEEP_OUTPUT_DIR / "all_runs_summary.csv")
    if failures:
        print("Failed runs saved to:", SWEEP_OUTPUT_DIR / "failed_runs.json")
    return summary_df


# ===== 一键开始整夜实验 =====
# 如果只想测试流程，可以先运行：run_reward_sweep(EXPERIMENT_PLAN.head(1))
summary_df = run_reward_sweep(EXPERIMENT_PLAN)
try:
    display(summary_df)
except NameError:
    print(summary_df)


# In[ ]:


# ===== 汇总表与论文候选图 =====
SUMMARY_CSV = SWEEP_OUTPUT_DIR / "all_runs_summary.csv"
EVAL_CSV = SWEEP_OUTPUT_DIR / "all_eval_history.csv"

summary_df = pd.read_csv(SUMMARY_CSV)
eval_df = pd.read_csv(EVAL_CSV) if EVAL_CSV.exists() else pd.DataFrame()

print("Loaded summaries:", len(summary_df))
try:
    display(summary_df[[
        "run_id", "draw_reward", "train_iters", "repeat", "seed",
        "final_random_non_loss_rate", "final_minimax_non_loss_rate",
        "final_minimax_first_non_loss_rate", "final_minimax_second_non_loss_rate",
        "final_minimax_losses", "elapsed_s"
    ]])
except NameError:
    print(summary_df)

final_500 = summary_df[summary_df["train_iters"] == 500].copy()
agg_500 = (
    final_500
    .groupby("draw_reward")
    .agg(
        runs=("run_id", "count"),
        minimax_non_loss_mean=("final_minimax_non_loss_rate", "mean"),
        minimax_non_loss_std=("final_minimax_non_loss_rate", "std"),
        minimax_second_non_loss_mean=("final_minimax_second_non_loss_rate", "mean"),
        minimax_losses_mean=("final_minimax_losses", "mean"),
        random_win_mean=("final_random_win_rate", "mean"),
    )
    .reset_index()
)
agg_500.to_csv(SWEEP_OUTPUT_DIR / "summary_500_by_draw_reward.csv", index=False)
print("\n500-iteration aggregate by draw_reward:")
try:
    display(agg_500)
except NameError:
    print(agg_500)


SHOW_SUMMARY_FIGURES = False  # 改成 True 可以在 notebook 里显示汇总候选图


def save_current_fig(name):
    path = FIGURE_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    if SHOW_SUMMARY_FIGURES:
        plt.show()
    else:
        plt.close()
    print("saved:", path)


plt.figure(figsize=(8, 4.8))
for draw_reward, part in final_500.groupby("draw_reward"):
    xs = np.full(len(part), draw_reward, dtype=float)
    plt.scatter(xs, part["final_minimax_non_loss_rate"] * 100, s=70, alpha=0.85, label=f"draw={draw_reward}")
means = final_500.groupby("draw_reward")["final_minimax_non_loss_rate"].mean() * 100
plt.plot(means.index, means.values, color="black", marker="o", linewidth=2, label="Mean")
plt.xticks(DRAW_REWARDS, [str(x) for x in DRAW_REWARDS])
plt.ylim(0, 105)
plt.xlabel("Draw reward")
plt.ylabel("Non-loss rate vs Minimax (%)")
plt.title("Effect of draw reward on PPO robustness after 500 iterations")
plt.grid(axis="y", alpha=0.25)
plt.legend()
save_current_fig("figure_1_500_minimax_non_loss_by_draw_reward.png")

plt.figure(figsize=(8, 4.8))
for train_iters, marker in [(500, "o"), (1000, "s")]:
    part = summary_df[summary_df["train_iters"] == train_iters]
    grouped = part.groupby("draw_reward")["final_minimax_non_loss_rate"].mean() * 100
    plt.plot(grouped.index, grouped.values, marker=marker, linewidth=2, label=f"{train_iters} iterations")
plt.xticks(DRAW_REWARDS, [str(x) for x in DRAW_REWARDS])
plt.ylim(0, 105)
plt.xlabel("Draw reward")
plt.ylabel("Final non-loss rate vs Minimax (%)")
plt.title("Final robustness under different draw rewards and training lengths")
plt.grid(axis="y", alpha=0.25)
plt.legend()
save_current_fig("figure_2_500_vs_1000_minimax_non_loss.png")

if not eval_df.empty:
    periodic = eval_df[
        (eval_df["eval_phase"] == "periodic")
        & (eval_df["opponent_name"] == "minimax")
        & (eval_df["train_iters"] == 500)
    ].copy()
    if not periodic.empty:
        curve = periodic.groupby(["draw_reward", "iteration"])["non_loss_rate"].mean().reset_index()
        plt.figure(figsize=(8, 4.8))
        for draw_reward, part in curve.groupby("draw_reward"):
            plt.plot(part["iteration"], part["non_loss_rate"] * 100, marker="o", linewidth=2, label=f"draw={draw_reward}")
        plt.ylim(0, 105)
        plt.xlabel("Training iteration")
        plt.ylabel("Non-loss rate vs Minimax (%)")
        plt.title("Learning dynamics against Minimax under different draw rewards")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        save_current_fig("figure_3_500_learning_curve_vs_minimax.png")

final_eval_df = pd.DataFrame()
if not eval_df.empty:
    final_eval_df = eval_df[eval_df["eval_phase"] == "final"].copy()


def plot_outcome_distribution_from_summary(opponent_prefix, opponent_label, filename):
    required = [
        f"final_{opponent_prefix}_wins",
        f"final_{opponent_prefix}_draws",
        f"final_{opponent_prefix}_losses",
    ]
    if not all(col in final_500.columns for col in required) or final_500.empty:
        return

    grouped = final_500.groupby("draw_reward")[required].mean().reindex(DRAW_REWARDS)
    totals = grouped.sum(axis=1).replace(0, np.nan)
    win_pct = grouped[f"final_{opponent_prefix}_wins"] / totals * 100
    draw_pct = grouped[f"final_{opponent_prefix}_draws"] / totals * 100
    loss_pct = grouped[f"final_{opponent_prefix}_losses"] / totals * 100

    x = np.arange(len(grouped.index))
    plt.figure(figsize=(8, 4.8))
    plt.bar(x, win_pct, label="Win")
    plt.bar(x, draw_pct, bottom=win_pct, label="Draw")
    plt.bar(x, loss_pct, bottom=win_pct + draw_pct, label="Loss")
    plt.xticks(x, [str(v) for v in grouped.index])
    plt.ylim(0, 105)
    plt.xlabel("Draw reward")
    plt.ylabel("Final outcome proportion (%)")
    plt.title(f"Final outcome distribution against {opponent_label} after 500 iterations")
    plt.legend()
    save_current_fig(filename)


# 图 4/5：胜/平/负分布，分别看 minimax 和 random
plot_outcome_distribution_from_summary(
    "minimax",
    "Minimax",
    "figure_4_500_minimax_final_outcome_distribution.png",
)
plot_outcome_distribution_from_summary(
    "random",
    "Random",
    "figure_5_500_random_final_outcome_distribution.png",
)

# 图 6：先手/后手不败率差异拆解，重点看 minimax
role_cols = ["final_minimax_first_non_loss_rate", "final_minimax_second_non_loss_rate"]
if not final_500.empty and all(col in final_500.columns for col in role_cols):
    role_grouped = final_500.groupby("draw_reward")[role_cols].mean().reindex(DRAW_REWARDS) * 100
    x = np.arange(len(role_grouped.index))
    width = 0.35
    plt.figure(figsize=(8, 4.8))
    plt.bar(x - width / 2, role_grouped["final_minimax_first_non_loss_rate"], width, label="PPO as first player")
    plt.bar(x + width / 2, role_grouped["final_minimax_second_non_loss_rate"], width, label="PPO as second player")
    plt.xticks(x, [str(v) for v in role_grouped.index])
    plt.ylim(0, 105)
    plt.xlabel("Draw reward")
    plt.ylabel("Non-loss rate vs Minimax (%)")
    plt.title("First-player and second-player robustness after 500 iterations")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    save_current_fig("figure_6_500_minimax_role_wise_non_loss.png")

# 图 7：对随机玩家的剥削能力，只看纯胜率
if not summary_df.empty and "final_random_win_rate" in summary_df.columns:
    plt.figure(figsize=(8, 4.8))
    for train_iters, marker in [(500, "o"), (1000, "s")]:
        part = summary_df[summary_df["train_iters"] == train_iters].copy()
        if part.empty:
            continue
        if train_iters == 500:
            for draw_reward, sub in part.groupby("draw_reward"):
                xs = np.full(len(sub), draw_reward, dtype=float)
                plt.scatter(xs, sub["final_random_win_rate"] * 100, s=65, alpha=0.8)
        grouped = part.groupby("draw_reward")["final_random_win_rate"].mean().reindex(DRAW_REWARDS) * 100
        plt.plot(grouped.index, grouped.values, marker=marker, linewidth=2, label=f"{train_iters} iterations")
    plt.xticks(DRAW_REWARDS, [str(x) for x in DRAW_REWARDS])
    plt.ylim(0, 105)
    plt.xlabel("Draw reward")
    plt.ylabel("Win rate vs Random (%)")
    plt.title("Exploitation ability against a random opponent")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    save_current_fig("figure_7_random_win_rate_by_draw_reward.png")

# 图 8/9：平均对局步数随训练轮数变化，分别看 minimax 和 random
if not eval_df.empty:
    periodic_steps = eval_df[
        (eval_df["eval_phase"] == "periodic")
        & (eval_df["train_iters"] == 500)
    ].copy()
    for opponent_name, filename, title_name in [
        ("minimax", "figure_8_500_avg_steps_vs_minimax.png", "Minimax"),
        ("random player", "figure_9_500_avg_steps_vs_random.png", "Random"),
    ]:
        part = periodic_steps[periodic_steps["opponent_name"] == opponent_name]
        if part.empty or "avg_steps" not in part.columns:
            continue
        curve = part.groupby(["draw_reward", "iteration"])["avg_steps"].mean().reset_index()
        plt.figure(figsize=(8, 4.8))
        for draw_reward, sub in curve.groupby("draw_reward"):
            plt.plot(sub["iteration"], sub["avg_steps"], marker="o", linewidth=2, label=f"draw={draw_reward}")
        plt.xlabel("Training iteration")
        plt.ylabel("Average game length (steps)")
        plt.title(f"Average game length against {title_name} during training")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        save_current_fig(filename)

SHOW_INDIVIDUAL_FIGURES = False  # 改成 True 可以在 notebook 里显示每个 run 的全部单独图
INDIVIDUAL_FIGURE_ROOT = FIGURE_DIR / "individual_runs"
INDIVIDUAL_FIGURE_ROOT.mkdir(parents=True, exist_ok=True)


def _save_individual_fig(run_fig_dir, name):
    path = run_fig_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    if SHOW_INDIVIDUAL_FIGURES:
        plt.show()
    else:
        plt.close()
    print("saved:", path)
    return str(path)


def _plot_if_available(ax, df, x_col, y_col, label, scale=1.0):
    if y_col in df.columns and df[y_col].notna().any():
        ax.plot(df[x_col], df[y_col] * scale, marker="o", linewidth=2, label=label)


def generate_individual_run_figures(run_id):
    run_dir = SWEEP_OUTPUT_DIR / run_id
    train_path = run_dir / "training_history.csv"
    eval_path = run_dir / "eval_history.csv"
    if not train_path.exists() or not eval_path.exists():
        print("skip individual figures, missing files:", run_id)
        return []

    train = pd.read_csv(train_path)
    ev = pd.read_csv(eval_path)
    periodic = ev[ev["eval_phase"] == "periodic"].copy()
    final = ev[ev["eval_phase"] == "final"].copy()
    run_fig_dir = INDIVIDUAL_FIGURE_ROOT / run_id
    saved = []

    if not train.empty:
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
        fig.suptitle(f"Training dynamics: {run_id}", fontsize=12)
        _plot_if_available(axes[0], train, "iteration", "episode_return_mean", "Episode return")
        axes[0].set_ylabel("Return")
        axes[0].grid(alpha=0.25)

        _plot_if_available(axes[1], train, "iteration", "episode_len_mean", "Episode length")
        axes[1].set_ylabel("Steps")
        axes[1].grid(alpha=0.25)

        if "entropy" in train.columns and train["entropy"].notna().any():
            _plot_if_available(axes[2], train, "iteration", "entropy", "Entropy")
            axes[2].set_ylabel("Entropy")
            axes[2].grid(alpha=0.25)
        else:
            axes[2].axis("off")
        axes[2].set_xlabel("Training iteration")
        for ax in axes:
            if ax.has_data():
                ax.legend()
        saved.append(_save_individual_fig(run_fig_dir, "figure_1_training_dynamics.png"))

    if not periodic.empty:
        plt.figure(figsize=(8, 4.8))
        for opponent_name, part in periodic.groupby("opponent_name"):
            plt.plot(part["iteration"], part["non_loss_rate"] * 100, marker="o", linewidth=2, label=opponent_name)
        plt.ylim(0, 105)
        plt.xlabel("Training iteration")
        plt.ylabel("Non-loss rate (%)")
        plt.title(f"Non-loss rate by opponent: {run_id}")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        saved.append(_save_individual_fig(run_fig_dir, "figure_2_non_loss_rate_by_opponent.png"))

        plt.figure(figsize=(8, 4.8))
        for opponent_name, part in periodic.groupby("opponent_name"):
            plt.plot(part["iteration"], part["loss_rate"] * 100, marker="o", linewidth=2, label=opponent_name)
        plt.ylim(0, 105)
        plt.xlabel("Training iteration")
        plt.ylabel("Loss rate (%)")
        plt.title(f"Loss rate by opponent: {run_id}")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        saved.append(_save_individual_fig(run_fig_dir, "figure_3_loss_rate_by_opponent.png"))

        plt.figure(figsize=(8, 4.8))
        for opponent_name, part in periodic.groupby("opponent_name"):
            plt.plot(part["iteration"], part["avg_steps"], marker="o", linewidth=2, label=opponent_name)
        plt.xlabel("Training iteration")
        plt.ylabel("Average game length (steps)")
        plt.title(f"Average game length by opponent: {run_id}")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        saved.append(_save_individual_fig(run_fig_dir, "figure_4_average_game_length_by_opponent.png"))

    for opponent_name, filename, title_name in [
        ("random player", "figure_5_evaluation_against_random.png", "Random"),
        ("minimax", "figure_6_evaluation_against_minimax.png", "Minimax"),
    ]:
        part = periodic[periodic["opponent_name"] == opponent_name].copy()
        if part.empty:
            continue
        plt.figure(figsize=(8, 4.8))
        plt.plot(part["iteration"], part["wins"], marker="o", linewidth=2, label="Wins")
        plt.plot(part["iteration"], part["draws"], marker="o", linewidth=2, label="Draws")
        plt.plot(part["iteration"], part["losses"], marker="o", linewidth=2, label="Losses")
        plt.xlabel("Training iteration")
        plt.ylabel("Games")
        plt.title(f"Evaluation against {title_name}: {run_id}")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        saved.append(_save_individual_fig(run_fig_dir, filename))

    if not final.empty:
        opponent_order = [x for x in ["random player", "minimax"] if x in final["opponent_name"].tolist()]
        final_plot = final.set_index("opponent_name").loc[opponent_order].reset_index()
        x = np.arange(len(final_plot))
        plt.figure(figsize=(7.5, 4.8))
        plt.bar(x, final_plot["wins"], label="Wins")
        plt.bar(x, final_plot["draws"], bottom=final_plot["wins"], label="Draws")
        plt.bar(x, final_plot["losses"], bottom=final_plot["wins"] + final_plot["draws"], label="Losses")
        plt.xticks(x, [name.replace(" player", "") for name in final_plot["opponent_name"]])
        plt.ylabel("Games")
        plt.title(f"Final evaluation outcomes: {run_id}")
        plt.legend()
        saved.append(_save_individual_fig(run_fig_dir, "figure_7_final_evaluation_outcomes.png"))

        width = 0.35
        plt.figure(figsize=(7.5, 4.8))
        plt.bar(x - width / 2, final_plot["first_non_loss_rate"] * 100, width, label="PPO as first player")
        plt.bar(x + width / 2, final_plot["second_non_loss_rate"] * 100, width, label="PPO as second player")
        plt.xticks(x, [name.replace(" player", "") for name in final_plot["opponent_name"]])
        plt.ylim(0, 105)
        plt.ylabel("Non-loss rate (%)")
        plt.title(f"Role-wise final non-loss rate: {run_id}")
        plt.grid(axis="y", alpha=0.25)
        plt.legend()
        saved.append(_save_individual_fig(run_fig_dir, "figure_8_role_wise_non_loss_rate.png"))

        final_plot.to_csv(run_fig_dir / "final_evaluation_table.csv", index=False)

    return saved


individual_index = []
for run_id in summary_df["run_id"]:
    saved_paths = generate_individual_run_figures(run_id)
    individual_index.append({
        "run_id": run_id,
        "figure_dir": str((INDIVIDUAL_FIGURE_ROOT / run_id).resolve()),
        "num_figures": len(saved_paths),
    })

individual_index_df = pd.DataFrame(individual_index)
individual_index_df.to_csv(FIGURE_DIR / "individual_run_figure_index.csv", index=False)
print("Individual run figures saved under:", INDIVIDUAL_FIGURE_ROOT.resolve())
try:
    display(individual_index_df)
except NameError:
    print(individual_index_df)

print("Figure directory:", FIGURE_DIR.absolute())


# ## 明天看结果时优先看哪些文件
# 
# 训练结束后，最重要的是这几个文件：
# 
# 1. `notebook_outputs/tictactoe_reward_sweep_overnight/all_runs_summary.csv`  
#    每个 run 的最终结果。
# 
# 2. `notebook_outputs/tictactoe_reward_sweep_overnight/summary_500_by_draw_reward.csv`  
#    500 轮三次重复后的均值和标准差，最适合写论文主表。
# 
# 3. `notebook_outputs/tictactoe_reward_sweep_overnight/all_eval_history.csv`  
#    每 50 轮评估一次的完整曲线数据。
# 
# 4. `paper_figures/tictactoe_reward_sweep_overnight/`  
#    自动生成的论文候选对比图。
# 
# 5. `paper_figures/tictactoe_reward_sweep_overnight/individual_runs/<run_id>/`  
#    每个 run 单独的一套图，结构接近原来的单实验 notebook，包括训练曲线、评估曲线、最终胜平负和先后手不败率。
# 
# 如果某个 run 中途失败，失败信息会保存到 `failed_runs.json`，已经完成的 run 不会丢失。再次运行 notebook 时，默认会跳过已完成的 run。
# 
