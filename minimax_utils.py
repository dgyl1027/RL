import random
import numpy as np
from functools import lru_cache

WIN_LINES = [
    [0, 1, 2],
    [3, 4, 5],
    [6, 7, 8],
    [0, 3, 6],
    [1, 4, 7],
    [2, 5, 8],
    [0, 4, 8],
    [2, 4, 6],
]


def check_win_board(board, mark):
    return any(all(board[i] == mark for i in line) for line in WIN_LINES)


def swap_perspective(board_tuple):
    swapped = []
    for x in board_tuple:
        if x == 1:
            swapped.append(2)
        elif x == 2:
            swapped.append(1)
        else:
            swapped.append(0)
    return tuple(swapped)


@lru_cache(maxsize=None)
def minimax_value(board_tuple):
    board = list(board_tuple)

    if check_win_board(board, 1):
        return 1
    if check_win_board(board, 2):
        return -1
    if 0 not in board:
        return 0

    best = -2
    legal = [i for i, x in enumerate(board) if x == 0]

    for a in legal:
        nxt = board[:]
        nxt[a] = 1

        if check_win_board(nxt, 1):
            return 1
        if 0 not in nxt:
            best = max(best, 0)
            continue

        child_board = swap_perspective(tuple(nxt))
        child_val = minimax_value(child_board)

        my_val = -child_val
        best = max(best, my_val)

        if best == 1:
            return 1

    return best


def minimax_action(obs):
    board = tuple(int(x) for x in obs["observation"])
    legal = np.flatnonzero(obs["action_mask"]).tolist()

    # 没有合法动作：通常说明这是终局/死回合。
    # 返回一个占位动作即可，环境会在 _was_dead_step 中忽略它。
    if len(legal) == 0:
        return 0

    best_actions = []
    best_val = -2

    for a in legal:
        nxt = list(board)
        nxt[a] = 1

        if check_win_board(nxt, 1):
            val = 1
        elif 0 not in nxt:
            val = 0
        else:
            child_board = swap_perspective(tuple(nxt))
            val = -minimax_value(child_board)

        if val > best_val:
            best_val = val
            best_actions = [a]
        elif val == best_val:
            best_actions.append(a)

    # 保险：理论上不会空，但为了健壮性再兜底一次
    if len(best_actions) == 0:
        return legal[0]

    return random.choice(best_actions)
