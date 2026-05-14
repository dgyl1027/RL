import json
from pathlib import Path


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


def write_notebook(path: str, cells: list[dict]) -> None:
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")


tictactoe_cells = [
    md(
        """
# Tic-Tac-Toe PPO 训练与论文图表生成

这个 notebook 用来重新训练 Tic-Tac-Toe，并自动保存论文需要的数据和图表。

建议先保持默认参数跑一次。如果时间不够，可以把 `TRAIN_ITERS` 调小；正式写论文时再使用完整轮数。
"""
    ),
    code(
        """
import os
import json
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import ray

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

import train_tictactoe as ttt
from tictactoe_env import TicTacToeEnv

print("GPU count:", ttt.torch.cuda.device_count())
print("CUDA available:", ttt.torch.cuda.is_available())
"""
    ),
    code(
        """
# ===== 可调参数 =====
TRAIN_ITERS = 300
EVAL_EVERY = 50
EVAL_GAMES_PER_EVAL = 500
FINAL_EVAL_GAMES = 1000

# 设为 1 可尝试 GPU；当前脚本默认 CPU，更容易复现。
NUM_GPUS = 0
NUM_ENV_RUNNERS = 2

OUTPUT_DIR = Path("notebook_outputs/tictactoe")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 使用 notebook 专用 checkpoint，避免覆盖原来的 checkpoints_tictactoe。
CHECKPOINT_DIR = Path("checkpoints_tictactoe_notebook").absolute()
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_HISTORY_CSV = OUTPUT_DIR / "training_history.csv"
EVAL_HISTORY_CSV = OUTPUT_DIR / "eval_history.csv"
FINAL_STATS_JSON = OUTPUT_DIR / "final_stats.json"

print("Output dir:", OUTPUT_DIR.absolute())
print("Checkpoint dir:", CHECKPOINT_DIR)
"""
    ),
    code(
        """
def build_tictactoe_config():
    env_name = "TicTacToe-v1-notebook"
    model_name = "tictactoe_mask_model_notebook"

    def env_creator(config):
        return TicTacToeEnv(config)

    register_env(env_name, env_creator)
    ModelCatalog.register_custom_model(model_name, ttt.TicTacToeMaskModel)

    temp_env = TicTacToeEnv()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env_name)
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
                [3000000, 0.0003],
                [6000000, 0.0],
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
            policy_mapping_fn=ttt.policy_mapping_fn,
            policies_to_train=[ttt.TRAIN_POLICY_ID],
        )
    )
    return config
"""
    ),
    code(
        """
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
    policy_rewards = env_stats.get("policy_reward_mean", {})
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


def flatten_eval_stats(stats, iteration, checkpoint_path=None):
    first = stats.get("first_stats", {})
    second = stats.get("second_stats", {})
    return {
        "iteration": iteration,
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
        "first_non_loss_rate": stats.get("first_non_loss_rate"),
        "second_win": second.get("win"),
        "second_draw": second.get("draw"),
        "second_loss": second.get("loss"),
        "second_non_loss_rate": stats.get("second_non_loss_rate"),
        "worst_role_non_loss_rate": stats.get("worst_role_non_loss_rate"),
    }


def save_histories(train_history, eval_history):
    pd.DataFrame(train_history).to_csv(TRAIN_HISTORY_CSV, index=False)
    pd.DataFrame(eval_history).to_csv(EVAL_HISTORY_CSV, index=False)
"""
    ),
    code(
        """
ray.shutdown()
ray.init(ignore_reinit_error=True, include_dashboard=False)

config = build_tictactoe_config()
algo = config.build_algo()

# 初始化历史策略池。
main_weights = algo.get_policy(ttt.TRAIN_POLICY_ID).get_weights()
algo.set_weights({pid: main_weights for pid in ttt.HISTORY_POLICY_IDS})

train_history = []
eval_history = []
best_score = -1.0
best_non_loss = -1.0
best_checkpoint_path = None
history_update_idx = 0

start_time = time.time()
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
        print(f"\\n===== Eval at iter {iteration} =====")
        stats = ttt.evaluate_against_random(algo, num_games=EVAL_GAMES_PER_EVAL)
        checkpoint_path = algo.save(str(CHECKPOINT_DIR))
        eval_history.append(flatten_eval_stats(stats, iteration, checkpoint_path))

        score = stats["worst_role_non_loss_rate"]
        non_loss = stats["non_loss_rate"]
        if score > best_score:
            best_score = score
            best_non_loss = non_loss
            best_checkpoint_path = checkpoint_path
            print("New best:", best_score, "overall_non_loss:", best_non_loss)

        # 更新历史池。
        target_hist_pid = ttt.HISTORY_POLICY_IDS[history_update_idx % ttt.HISTORY_POOL_SIZE]
        main_weights = algo.get_policy(ttt.TRAIN_POLICY_ID).get_weights()
        algo.set_weights({target_hist_pid: main_weights})
        history_update_idx += 1

        save_histories(train_history, eval_history)

final_checkpoint = algo.save(str(CHECKPOINT_DIR))
print("Final checkpoint:", final_checkpoint)
print("Best checkpoint:", best_checkpoint_path)
print("Elapsed seconds:", time.time() - start_time)

if best_checkpoint_path is not None:
    best_algo = config.build_algo()
    best_algo.restore(best_checkpoint_path)
    final_stats = ttt.evaluate_against_random(best_algo, num_games=FINAL_EVAL_GAMES)
    best_algo.stop()
else:
    final_stats = {}

with open(FINAL_STATS_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "final_checkpoint": str(final_checkpoint),
            "best_checkpoint": str(best_checkpoint_path),
            "best_score": best_score,
            "best_non_loss": best_non_loss,
            "final_stats": final_stats,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

save_histories(train_history, eval_history)
algo.stop()
ray.shutdown()
"""
    ),
    code(
        """
train_df = pd.read_csv(TRAIN_HISTORY_CSV)
eval_df = pd.read_csv(EVAL_HISTORY_CSV) if EVAL_HISTORY_CSV.exists() else pd.DataFrame()

def save_current_fig(name):
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    print("saved:", path)
    plt.show()

# 图 1：训练过程曲线
cols = [
    "episode_return_mean",
    "episode_len_mean",
    "main_policy_reward_mean",
    "total_loss",
    "entropy",
]
cols = [c for c in cols if c in train_df.columns and train_df[c].notna().any()]
fig, axes = plt.subplots(len(cols), 1, figsize=(10, 3 * len(cols)), sharex=True)
if len(cols) == 1:
    axes = [axes]
for ax, col in zip(axes, cols):
    ax.plot(train_df["iteration"], train_df[col], alpha=0.45, label=col)
    ax.plot(
        train_df["iteration"],
        train_df[col].rolling(15, min_periods=1).mean(),
        linewidth=2,
        label=f"{col} rolling mean",
    )
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    ax.legend()
axes[-1].set_xlabel("Training iteration")
save_current_fig("ttt_training_curves.png")

# 图 2：定期评估曲线
if not eval_df.empty:
    plt.figure(figsize=(10, 5))
    for col in ["win_rate", "non_loss_rate", "first_non_loss_rate", "second_non_loss_rate", "worst_role_non_loss_rate"]:
        if col in eval_df.columns:
            plt.plot(eval_df["iteration"], eval_df[col], marker="o", label=col)
    plt.ylim(0, 1.05)
    plt.xlabel("Training iteration")
    plt.ylabel("Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current_fig("ttt_eval_rates.png")

    # 图 3：最后一次评估的胜/平/负
    last = eval_df.iloc[-1]
    plt.figure(figsize=(6, 4))
    plt.bar(["Win", "Draw", "Loss"], [last.get("wins", 0), last.get("draws", 0), last.get("losses", 0)])
    plt.ylabel("Games")
    plt.title("Tic-Tac-Toe final evaluation outcomes")
    save_current_fig("ttt_final_outcomes.png")

    # 图 4：先手/后手不败率
    plt.figure(figsize=(6, 4))
    plt.bar(["First player", "Second player"], [last.get("first_non_loss_rate", 0), last.get("second_non_loss_rate", 0)])
    plt.ylim(0, 1.05)
    plt.ylabel("Non-loss rate")
    plt.title("Role-wise non-loss rate")
    save_current_fig("ttt_role_non_loss.png")
"""
    ),
    md(
        """
## 建议放入论文的 Tic-Tac-Toe 图表

1. `ttt_training_curves.png`：训练过程中的 return、episode length、main policy reward、loss/entropy。
2. `ttt_eval_rates.png`：每 50 轮评估一次的不败率、胜率、先后手表现。
3. `ttt_final_outcomes.png`：最终评估胜/平/负柱状图。
4. `ttt_role_non_loss.png`：先手和后手不败率比较。
"""
    ),
]


einstein_cells = [
    md(
        """
# Einstein Wuerfelt Nicht! PPO 训练与论文图表生成

这个 notebook 用来重新训练 Einstein，并自动保存训练曲线、评估数据、结束原因统计和论文图表。
"""
    ),
    code(
        """
import os
import json
import time
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import ray

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec

import train_einstein as ein
from einstein_env import EinsteinEnv

print("GPU count:", ein.torch.cuda.device_count())
print("CUDA available:", ein.torch.cuda.is_available())
"""
    ),
    code(
        """
# ===== 可调参数 =====
TRAIN_ITERS = 1000
EVAL_EVERY = 50
EVAL_GAMES_PER_EVAL = 500
FINAL_EVAL_GAMES = 1000

NUM_GPUS = 0
NUM_ENV_RUNNERS = 2

ENV_CONFIG = dict(ein.ENV_CONFIG)

OUTPUT_DIR = Path("notebook_outputs/einstein")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 使用 notebook 专用 checkpoint，避免覆盖原来的 checkpoints_einstein_setup。
CHECKPOINT_DIR = Path("checkpoints_einstein_notebook").absolute()
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_HISTORY_CSV = OUTPUT_DIR / "training_history.csv"
EVAL_HISTORY_CSV = OUTPUT_DIR / "eval_history.csv"
REASON_HISTORY_CSV = OUTPUT_DIR / "reason_history.csv"
FINAL_STATS_JSON = OUTPUT_DIR / "final_stats.json"

print("ENV_CONFIG:", ENV_CONFIG)
print("Output dir:", OUTPUT_DIR.absolute())
print("Checkpoint dir:", CHECKPOINT_DIR)
"""
    ),
    code(
        """
def build_einstein_config():
    env_name = "einstein_env_notebook"
    model_name = "einstein_action_mask_model_notebook"

    def env_creator(config):
        return EinsteinEnv(config)

    register_env(env_name, env_creator)
    ModelCatalog.register_custom_model(model_name, ein.EinsteinActionMaskModel)

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
            num_env_runners=NUM_ENV_RUNNERS,
            num_gpus_per_env_runner=0,
        )
        .resources(num_gpus=NUM_GPUS)
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
                ein.TRAIN_POLICY_ID: PolicySpec(None, obs_space, act_space, {}),
                ein.RANDOM_POLICY_ID: PolicySpec(
                    policy_class=ein.RandomMaskedPolicy,
                    observation_space=obs_space,
                    action_space=act_space,
                    config={},
                ),
                ein.HEURISTIC_POLICY_ID: PolicySpec(
                    policy_class=ein.HeuristicMaskedPolicy,
                    observation_space=obs_space,
                    action_space=act_space,
                    config={},
                ),
                **{
                    pid: PolicySpec(None, obs_space, act_space, {})
                    for pid in ein.HISTORY_POLICY_IDS
                },
            },
            policy_mapping_fn=ein.policy_mapping_fn,
            policies_to_train=[ein.TRAIN_POLICY_ID],
        )
    )
    return config
"""
    ),
    code(
        """
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
        "main_policy_reward_mean": policy_rewards.get(ein.TRAIN_POLICY_ID),
        "random_policy_reward_mean": policy_rewards.get(ein.RANDOM_POLICY_ID),
        "heuristic_policy_reward_mean": policy_rewards.get(ein.HEURISTIC_POLICY_ID),
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


def save_histories(train_history, eval_history, reason_history):
    pd.DataFrame(train_history).to_csv(TRAIN_HISTORY_CSV, index=False)
    pd.DataFrame(eval_history).to_csv(EVAL_HISTORY_CSV, index=False)
    pd.DataFrame(reason_history).to_csv(REASON_HISTORY_CSV, index=False)
"""
    ),
    code(
        """
ray.shutdown()
ray.init(ignore_reinit_error=True, include_dashboard=False)

config = build_einstein_config()
algo = config.build_algo()

main_weights = algo.get_policy(ein.TRAIN_POLICY_ID).get_weights()
algo.set_weights({pid: main_weights for pid in ein.HISTORY_POLICY_IDS})

train_history = []
eval_history = []
reason_history = []
best_score = -1.0
best_win_rate = -1.0
best_checkpoint_path = None
history_update_idx = 0

start_time = time.time()
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
        print(f"\\n===== Eval at iter {iteration} =====")
        random_stats = ein.evaluate_against_random(algo, num_games=EVAL_GAMES_PER_EVAL)
        heuristic_stats = ein.evaluate_against_heuristic(algo, num_games=EVAL_GAMES_PER_EVAL)
        checkpoint_dir = CHECKPOINT_DIR / f"iter_{iteration:04d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = algo.save(str(checkpoint_dir))
        eval_history.append(flatten_eval_stats(random_stats, iteration, checkpoint_path))
        eval_history.append(flatten_eval_stats(heuristic_stats, iteration, checkpoint_path))
        reason_history.extend(reason_rows(random_stats, iteration))
        reason_history.extend(reason_rows(heuristic_stats, iteration))

        score = min(
            random_stats["worst_role_win_rate"],
            heuristic_stats["worst_role_win_rate"],
        )
        if score > best_score:
            best_score = score
            best_win_rate = min(random_stats["win_rate"], heuristic_stats["win_rate"])
            best_checkpoint_path = checkpoint_path
            print("New best:", best_score, "overall_win_rate:", best_win_rate)

        target_hist_pid = ein.HISTORY_POLICY_IDS[history_update_idx % ein.HISTORY_POOL_SIZE]
        main_weights = algo.get_policy(ein.TRAIN_POLICY_ID).get_weights()
        algo.set_weights({target_hist_pid: main_weights})
        history_update_idx += 1

        save_histories(train_history, eval_history, reason_history)

final_checkpoint_dir = CHECKPOINT_DIR / "final"
final_checkpoint_dir.mkdir(parents=True, exist_ok=True)
final_checkpoint = algo.save(str(final_checkpoint_dir))
print("Final checkpoint:", final_checkpoint)
print("Best checkpoint:", best_checkpoint_path)
print("Elapsed seconds:", time.time() - start_time)

if best_checkpoint_path is not None:
    best_algo = config.build_algo()
    best_algo.restore(best_checkpoint_path)
    final_stats = {
        "random_player": ein.evaluate_against_random(best_algo, num_games=FINAL_EVAL_GAMES),
        "heuristic_player": ein.evaluate_against_heuristic(best_algo, num_games=FINAL_EVAL_GAMES),
    }
    best_algo.stop()
else:
    final_stats = {}

with open(FINAL_STATS_JSON, "w", encoding="utf-8") as f:
    json.dump(
        {
            "env_config": ENV_CONFIG,
            "final_checkpoint": str(final_checkpoint),
            "best_checkpoint": str(best_checkpoint_path),
            "best_score": best_score,
            "best_win_rate": best_win_rate,
            "final_stats": final_stats,
        },
        f,
        ensure_ascii=False,
        indent=2,
    )

save_histories(train_history, eval_history, reason_history)
algo.stop()
ray.shutdown()
"""
    ),
    code(
        """
train_df = pd.read_csv(TRAIN_HISTORY_CSV)
eval_df = pd.read_csv(EVAL_HISTORY_CSV) if EVAL_HISTORY_CSV.exists() else pd.DataFrame()
reason_df = pd.read_csv(REASON_HISTORY_CSV) if REASON_HISTORY_CSV.exists() else pd.DataFrame()

def save_current_fig(name):
    path = OUTPUT_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    print("saved:", path)
    plt.show()

# 图 1：训练过程曲线
cols = [
    "episode_return_mean",
    "episode_len_mean",
    "main_policy_reward_mean",
    "total_loss",
    "entropy",
]
cols = [c for c in cols if c in train_df.columns and train_df[c].notna().any()]
fig, axes = plt.subplots(len(cols), 1, figsize=(10, 3 * len(cols)), sharex=True)
if len(cols) == 1:
    axes = [axes]
for ax, col in zip(axes, cols):
    ax.plot(train_df["iteration"], train_df[col], alpha=0.45, label=col)
    ax.plot(
        train_df["iteration"],
        train_df[col].rolling(15, min_periods=1).mean(),
        linewidth=2,
        label=f"{col} rolling mean",
    )
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    ax.legend()
axes[-1].set_xlabel("Training iteration")
save_current_fig("einstein_training_curves.png")

# 图 2：定期评估曲线
if not eval_df.empty:
    plt.figure(figsize=(10, 5))
    metric_cols = ["win_rate", "worst_role_win_rate", "first_win_rate", "second_win_rate"]
    if "opponent_name" in eval_df.columns:
        for opponent_name, sub_df in eval_df.groupby("opponent_name"):
            for col in metric_cols:
                if col in sub_df.columns:
                    plt.plot(
                        sub_df["iteration"],
                        sub_df[col],
                        marker="o",
                        label=f"{opponent_name} {col}",
                    )
    else:
        for col in metric_cols:
            if col in eval_df.columns:
                plt.plot(eval_df["iteration"], eval_df[col], marker="o", label=col)
    plt.ylim(0, 1.05)
    plt.xlabel("Training iteration")
    plt.ylabel("Rate")
    plt.grid(True, alpha=0.3)
    plt.legend()
    save_current_fig("einstein_eval_rates.png")

    # 图 3：最后一次评估的胜/平/负
    last = eval_df.iloc[-1]
    plt.figure(figsize=(6, 4))
    plt.bar(["Win", "Draw/Truncated", "Loss"], [last.get("wins", 0), last.get("draws", 0), last.get("losses", 0)])
    plt.ylabel("Games")
    plt.title("Einstein final evaluation outcomes")
    save_current_fig("einstein_final_outcomes.png")

    # 图 4：先手/后手胜率
    plt.figure(figsize=(6, 4))
    plt.bar(["First player", "Second player"], [last.get("first_win_rate", 0), last.get("second_win_rate", 0)])
    plt.ylim(0, 1.05)
    plt.ylabel("Win rate")
    plt.title("Role-wise win rate")
    save_current_fig("einstein_role_win_rate.png")

# 图 5：结束原因统计
if not reason_df.empty:
    final_iter = reason_df["iteration"].max()
    sub = reason_df[reason_df["iteration"] == final_iter].sort_values("count", ascending=False)
    plt.figure(figsize=(8, 4))
    plt.bar(sub["reason"], sub["count"])
    plt.ylabel("Games")
    plt.title("Einstein final evaluation end reasons")
    plt.xticks(rotation=30, ha="right")
    save_current_fig("einstein_end_reasons.png")
"""
    ),
    md(
        """
## 建议放入论文的 Einstein 图表

1. `einstein_training_curves.png`：训练过程中的 return、episode length、main policy reward、loss/entropy。
2. `einstein_eval_rates.png`：每 50 轮评估一次的胜率、不败率、先后手表现。
3. `einstein_final_outcomes.png`：最终评估胜/平或截断/负柱状图。
4. `einstein_role_win_rate.png`：先手和后手胜率比较。
5. `einstein_end_reasons.png`：终局原因统计，例如 goal、capture_all、no_legal_action、truncated。
"""
    ),
]


if __name__ == "__main__":
    write_notebook("train_tictactoe_experiment.ipynb", tictactoe_cells)
    write_notebook("train_einstein_experiment.ipynb", einstein_cells)
    print("Wrote train_tictactoe_experiment.ipynb")
    print("Wrote train_einstein_experiment.ipynb")
