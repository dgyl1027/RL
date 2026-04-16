import os
import random
from functools import lru_cache

import numpy as np
import ray

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import Policy, PolicySpec

from tictactoe_env import TicTacToeEnv

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

from minimax_utils import minimax_action

torch, nn = try_import_torch()

TRAIN_POLICY_ID = "main_policy"
RANDOM_POLICY_ID = "random_policy"
MINIMAX_POLICY_ID = "minimax_policy"

HISTORY_POOL_SIZE = 3
HISTORY_POLICY_IDS = [f"hist_policy_{i}" for i in range(HISTORY_POOL_SIZE)]
MAIN_POLICY_SECOND_BUCKETS = 7

print("========================================")
print(f"检测到 GPU 数量: {torch.cuda.device_count()}")
print(f"PyTorch CUDA 是否可用: {torch.cuda.is_available()}")
print("========================================")


# ==========================================
# 1. 自定义动作掩码模型
# ==========================================
class TicTacToeMaskModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # 重新应用：One-Hot通道分离
        self.fc1 = nn.Linear(18, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_branch = nn.Linear(128, 9)
        self.value_branch = nn.Linear(128, 1)

        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        board = input_dict["obs"]["observation"].float()
        action_mask = input_dict["obs"]["action_mask"].float()

        # 重新应用：将 1 和 2 分离，使得多层感知机可以轻易区分敌我
        my_pieces = (board == 1.0).float()
        opp_pieces = (board == 2.0).float()
        encoded_board = torch.cat([my_pieces, opp_pieces], dim=-1)

        x = torch.relu(self.fc1(encoded_board))
        x = torch.relu(self.fc2(x))

        logits = self.action_branch(x)
        self._value_out = self.value_branch(x).squeeze(1)

        safe_mask = (1.0 - action_mask) * -1e9
        masked_logits = logits + safe_mask
        return masked_logits, state

    def value_function(self):
        return self._value_out


# ==========================================
# 2. 固定对手策略：random / minimax
# ==========================================


def unpack_obs_batch(obs_batch):
    """
    兼容 RLlib old API stack 下固定策略收到的多种 obs_batch 格式：
    1) dict of arrays:
       {"observation": [B,9], "action_mask": [B,9]}
    2) list[dict] / tuple[dict]
    3) np.ndarray(dtype=object)，每个元素是 dict
    4) np.ndarray(dtype=float32/float64)，形状可能是:
       - [18]      : 单个样本扁平化后
       - [B, 18]   : 一个 batch 扁平化后
       RLlib 扁平化 Dict 时顺序可能随接口变化，这里会自动识别哪半边是
       observation，哪半边是 action_mask
    """
    # 情况 1：dict of arrays
    if isinstance(obs_batch, dict):
        batch_size = len(obs_batch["action_mask"])
        out = []
        for i in range(batch_size):
            out.append(
                {
                    "observation": np.asarray(obs_batch["observation"][i]),
                    "action_mask": np.asarray(obs_batch["action_mask"][i]),
                }
            )
        return out

    # 情况 2：list / tuple
    if isinstance(obs_batch, (list, tuple)):
        return list(obs_batch)

    # 情况 3/4：numpy 数组
    if isinstance(obs_batch, np.ndarray):
        # 3) object 数组，里面每个元素是 dict
        if obs_batch.dtype == object:
            return list(obs_batch)

        # 4) 扁平 float 数组
        if np.issubdtype(obs_batch.dtype, np.floating) or np.issubdtype(
            obs_batch.dtype, np.integer
        ):
            arr = np.asarray(obs_batch)

            # 单个样本：[18]
            if arr.ndim == 1:
                if arr.shape[0] != 18:
                    raise TypeError(f"Unexpected 1D obs_batch shape: {arr.shape}")
                arr = arr.reshape(1, 18)

            # batch 样本：[B, 18]
            if arr.ndim == 2:
                if arr.shape[1] != 18:
                    raise TypeError(f"Unexpected 2D obs_batch shape: {arr.shape}")

                return [_split_flat_obs(arr[i]) for i in range(arr.shape[0])]

            raise TypeError(
                f"Unexpected ndarray obs_batch ndim: {arr.ndim}, shape={arr.shape}"
            )

    raise TypeError(f"Unsupported obs_batch type: {type(obs_batch)}")


def _valid_obs_mask_pair(obs, mask):
    obs = np.asarray(obs)
    mask = np.asarray(mask)

    if obs.shape != (9,) or mask.shape != (9,):
        return False
    if not np.all(np.isin(obs, [0, 1, 2])):
        return False
    if not np.all(np.isin(mask, [0, 1])):
        return False

    return np.array_equal(mask.astype(np.int8), (obs == 0).astype(np.int8))


def _split_flat_obs(flat):
    first = np.asarray(flat[:9]).astype(np.int8)
    second = np.asarray(flat[9:]).astype(np.int8)

    # RLlib old stack 常见是 Dict 按 key 排序后扁平化：action_mask 在前。
    if _valid_obs_mask_pair(second, first):
        return {"observation": second, "action_mask": first}

    if _valid_obs_mask_pair(first, second):
        return {"observation": first, "action_mask": second}

    raise TypeError(
        "Unable to split flattened observation into observation/action_mask: "
        f"first={first.tolist()}, second={second.tolist()}"
    )


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

            if len(legal) == 0:
                actions.append(0)
            else:
                actions.append(random.choice(legal))

        return np.array(actions, dtype=np.int64), [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass


class MinimaxPolicy(Policy):
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
            single_obs = {
                "observation": np.asarray(obs["observation"]),
                "action_mask": np.asarray(obs["action_mask"]),
            }
            actions.append(minimax_action(single_obs))

        return np.array(actions, dtype=np.int64), [], {}

    def learn_on_batch(self, samples):
        return {}

    def get_weights(self):
        return {}

    def set_weights(self, weights):
        pass


# ==========================================
# 3. 一些工具函数
# ==========================================
def find_metric(d, key):
    if key in d:
        return d[key]
    for _, v in d.items():
        if isinstance(v, dict):
            res = find_metric(v, key)
            if res is not None:
                return res
    return None


def get_episode_hash(episode):
    raw_episode_id = getattr(episode, "id_", None)

    if raw_episode_id is None:
        raw_episode_id = getattr(episode, "episode_id", None)

    if raw_episode_id is None:
        raw_episode_id = str(episode)

    return hash(str(raw_episode_id))


def choose_opponent_policy(episode_hash, main_as_second=False):
    """
    训练对手分布：
    - main_policy 后手时：更高比例对 minimax 先手，专门练防守和补洞。
    - main_policy 先手时：保留更多 random / 历史池，继续学习抓随机玩家破绽。
    """
    bucket = episode_hash % 10

    if main_as_second:
        if bucket < 7:
            return MINIMAX_POLICY_ID
        if bucket < 9:
            return RANDOM_POLICY_ID

        hist_idx = (episode_hash // 10) % HISTORY_POOL_SIZE
        return HISTORY_POLICY_IDS[hist_idx]

    if bucket < 4:
        hist_idx = (episode_hash // 10) % HISTORY_POOL_SIZE
        return HISTORY_POLICY_IDS[hist_idx]

    if bucket < 8:
        return RANDOM_POLICY_ID

    return MINIMAX_POLICY_ID


def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    """
    约 70% 局里 main_policy 后手，约 30% 局里 main_policy 先手。
    对手从：历史池 / random / minimax 中抽取
    """
    episode_hash = get_episode_hash(episode)
    main_as_second = episode_hash % 10 < MAIN_POLICY_SECOND_BUCKETS
    opp_policy_id = choose_opponent_policy(episode_hash, main_as_second)

    if main_as_second:
        return TRAIN_POLICY_ID if agent_id == "player_2" else opp_policy_id

    return TRAIN_POLICY_ID if agent_id == "player_1" else opp_policy_id


def summarize_role_stats(stats):
    total = sum(stats.values())
    if total == 0:
        return {
            "total": 0,
            "win_rate": 0.0,
            "non_loss_rate": 0.0,
        }

    return {
        "total": total,
        "win_rate": stats["win"] / total,
        "non_loss_rate": (stats["win"] + stats["draw"]) / total,
    }


def evaluate_against_random(algo, num_games=200):
    first_stats = {"win": 0, "draw": 0, "loss": 0}
    second_stats = {"win": 0, "draw": 0, "loss": 0}
    step_list = []

    for game_idx in range(num_games):
        env = TicTacToeEnv()
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
                action = algo.compute_single_action(
                    current_obs,
                    policy_id=TRAIN_POLICY_ID,
                    explore=False,
                )
                if isinstance(action, tuple):
                    action = action[0]
                action = int(action)
            else:
                legal_actions = np.flatnonzero(current_obs["action_mask"]).tolist()
                action = random.choice(legal_actions) if legal_actions else 0

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
    avg_steps = sum(step_list) / len(step_list) if step_list else 0.0
    first_summary = summarize_role_stats(first_stats)
    second_summary = summarize_role_stats(second_stats)
    non_loss_rate = (win + draw) / num_games
    worst_role_non_loss_rate = min(
        first_summary["non_loss_rate"], second_summary["non_loss_rate"]
    )

    print("\n================ 评估结果：对随机玩家 ================")
    print(f"总对局数: {num_games}")
    print(f"胜: {win} | 平: {draw} | 负: {loss}")
    print(f"胜率: {win / num_games:.3f}")
    print(f"不败率: {non_loss_rate:.3f}")
    print(f"先后手最弱不败率: {worst_role_non_loss_rate:.3f}")
    print(f"平均步数: {avg_steps:.2f}")

    print("\n--- 模型先手(player_1) ---")
    if first_summary["total"] > 0:
        print(
            f"胜: {first_stats['win']} | 平: {first_stats['draw']} | 负: {first_stats['loss']} "
            f"| 胜率: {first_summary['win_rate']:.3f} "
            f"| 不败率: {first_summary['non_loss_rate']:.3f}"
        )

    print("\n--- 模型后手(player_2) ---")
    if second_summary["total"] > 0:
        print(
            f"胜: {second_stats['win']} | 平: {second_stats['draw']} | 负: {second_stats['loss']} "
            f"| 胜率: {second_summary['win_rate']:.3f} "
            f"| 不败率: {second_summary['non_loss_rate']:.3f}"
        )

    print("====================================================\n")
    return {
        "wins": win,
        "draws": draw,
        "losses": loss,
        "avg_steps": avg_steps,
        "first_stats": first_stats,
        "second_stats": second_stats,
        "first_non_loss_rate": first_summary["non_loss_rate"],
        "second_non_loss_rate": second_summary["non_loss_rate"],
        "worst_role_non_loss_rate": worst_role_non_loss_rate,
        "non_loss_rate": non_loss_rate,
        "win_rate": win / num_games,
    }


# ==========================================
# 4. 训练主流程
# ==========================================
if __name__ == "__main__":
    TRAIN_ITERS = 300
    EVAL_GAMES = 1000
    CHECKPOINT_DIR = os.path.abspath("./checkpoints_tictactoe")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"checkpoint 保存目录: {CHECKPOINT_DIR}")

    ray.init(ignore_reinit_error=True)

    # 注册环境
    def env_creator(config):
        return TicTacToeEnv(config)

    register_env("TicTacToe-v1", env_creator)

    # 注册模型
    ModelCatalog.register_custom_model("my_mask_model", TicTacToeMaskModel)

    temp_env = TicTacToeEnv()
    obs_space = temp_env.observation_space
    act_space = temp_env.action_space

    # PPO 配置
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment("TicTacToe-v1")
        .framework("torch")
        .resources(
            num_gpus=0,
        )
        .env_runners(
            num_env_runners=2,
            num_gpus_per_env_runner=0,
        )
        .training(
            model={"custom_model": "my_mask_model"},
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
                TRAIN_POLICY_ID: PolicySpec(None, obs_space, act_space, {}),
                RANDOM_POLICY_ID: PolicySpec(
                    policy_class=RandomMaskedPolicy,
                    observation_space=obs_space,
                    action_space=act_space,
                    config={},
                ),
                MINIMAX_POLICY_ID: PolicySpec(
                    policy_class=MinimaxPolicy,
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
    print("开始训练井字棋 PPO(混合对手版)...")

    # 初始化历史池：复制 main_policy 的初始参数并广播给所有 workers
    main_weights = algo.get_policy(TRAIN_POLICY_ID).get_weights()
    algo.set_weights({pid: main_weights for pid in HISTORY_POLICY_IDS})

    best_score = -1.0
    best_non_loss = -1.0
    best_checkpoint_path = None

    history_update_idx = 0

    for i in range(TRAIN_ITERS):
        result = algo.train()

        if (i + 1) % 10 == 0:
            print(f"第 {i + 1:03d} 轮")

        if (i + 1) % 50 == 0:
            print(f"\n===== 第 {i + 1} 轮后开始评估 =====")
            stats = evaluate_against_random(algo, num_games=500)

            checkpoint_path = algo.save(CHECKPOINT_DIR)
            print(f"已保存 checkpoint: {checkpoint_path}")

            score = stats["worst_role_non_loss_rate"]
            non_loss = stats["non_loss_rate"]
            if score > best_score:
                best_score = score
                best_non_loss = non_loss
                best_checkpoint_path = checkpoint_path
                print(
                    "新的最佳模型！"
                    f"worst_role_non_loss_rate = {best_score:.3f} | "
                    f"overall_non_loss_rate = {best_non_loss:.3f}"
                )

        # 每 50 轮更新一个历史槽位，形成小型历史池并全局广播
        if (i + 1) % 50 == 0:
            target_hist_pid = HISTORY_POLICY_IDS[history_update_idx % HISTORY_POOL_SIZE]
            main_weights = algo.get_policy(TRAIN_POLICY_ID).get_weights()
            algo.set_weights({target_hist_pid: main_weights})
            print(f"已将 main_policy 同步到 {target_hist_pid}")
            history_update_idx += 1

    # 训练结束后保存最终模型
    final_checkpoint = algo.save(CHECKPOINT_DIR)
    print(f"\n最终 checkpoint 已保存: {final_checkpoint}")

    # 打印训练过程中找到的最佳模型
    print(f"\n最佳 checkpoint: {best_checkpoint_path}")
    print(f"最佳整体不败率: {best_non_loss:.3f}")
    print(f"最佳先后手最弱不败率: {best_score:.3f}")

    algo.stop()

    if best_checkpoint_path is None:
        print("没有找到最佳 checkpoint，跳过最终评估。")
        ray.shutdown()
    else:
        best_algo = config.build_algo()
        best_algo.restore(best_checkpoint_path)

        final_stats = evaluate_against_random(best_algo, num_games=EVAL_GAMES)
        print(f"最佳模型最终不败率: {final_stats['non_loss_rate']:.3f}")
        print(
            "最佳模型最终先后手最弱不败率: "
            f"{final_stats['worst_role_non_loss_rate']:.3f}"
        )

        best_algo.stop()
        ray.shutdown()
