import numpy as np
from matplotlib import pyplot as plt

from constant import *
from settings import *

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def get_drink_concentration():
    if not guzzling_pouch:
        return 1.0
    return 1 + 0.1 * (1 + BONUS[guzzling_pouch_level])


def get_player_level():
    return base_player_level + get_drink_concentration() * (
            (3 if tea['enhancing tea'] else 0) +
            (6 if tea['super enhancing tea'] else 0) +
            (8 if tea['ultra enhancing tea'] else 0)
    )


def get_success_rate_mod(item_recommended_level: int):
    player_level = get_player_level()
    level_rate = ((player_level - item_recommended_level) * 0.0005) \
        if (player_level >= item_recommended_level) \
        else (-0.5 * (1 - player_level / item_recommended_level))
    enhancer_buff = TOOLS_LEVEL[enhancer_type] * (1 + BONUS[enhancer_level])
    enhancer_buff = round(enhancer_buff, 4)  # doh-nuts website has this rounding, and I don't know why
    buff = 0.0005 * laboratory_level + enhancer_buff
    return level_rate + buff


def get_success_rate(item_recommended_level: int, item_level: int):
    return BASE_SUCCESS_RATE[item_level] * (1 + get_success_rate_mod(item_recommended_level))


def solve(protection_level: int):
    """
    Math theorem:
    Let n = target level
    Let p_k = enhance success rate when you have a +k item
    Let x_k = expect of enhance times of getting the target item when you have a +k item
    Do not account protection or blessed tea
    x_k = 1 + (1-p)*x_0 + p*x_{k+1}
    (p-1)*x_0 - p*x_{k+1} + x_k = 1     (k th equation)
    Combine as n+1 equations, i.e. a system of linear equations AX = b
    Trivial to consider protection or blessed tea
    :return expect of enhance times when you have a +0 item
    """
    matrix = np.zeros((target_level + 1, target_level + 1))
    blessed_rate = 0.01 * get_drink_concentration() if tea['blessed tea'] else 0.0
    for now_level in range(target_level + 1):
        success_rate = get_success_rate(recommended_level, now_level)
        if now_level == target_level:
            matrix[now_level][now_level] = 1.0
            continue
        matrix[now_level][now_level] = 1.0
        if now_level < protection_level:
            matrix[now_level][0] += success_rate - 1
        else:
            matrix[now_level][now_level - 1] = success_rate - 1
        if now_level + 1 < target_level:
            matrix[now_level][now_level + 1] = -success_rate * (1 - blessed_rate)
            matrix[now_level][now_level + 2] = -success_rate * blessed_rate
        else:
            matrix[now_level][now_level + 1] = -success_rate
    return np.linalg.solve(matrix, np.ones(target_level + 1))[0]


def stochastic_matrix(protection_level: int):
    """
    matrix[i][j] = when you have a +i item, matrix[i][j] possibility to get a +j item after one action
    """
    matrix = np.zeros((target_level + 1, target_level + 1))
    blessed_rate = 0.01 * get_drink_concentration() if tea['blessed tea'] else 0.0
    for now_level in range(target_level + 1):
        if now_level == target_level:
            matrix[now_level][now_level] = 1.0
            continue
        success_rate = get_success_rate(recommended_level, now_level)
        # success
        if now_level + 1 < target_level:
            matrix[now_level][now_level + 1] = success_rate * (1 - blessed_rate)
            matrix[now_level][now_level + 2] = success_rate * blessed_rate
        else:
            matrix[now_level][now_level + 1] = success_rate
        # fail
        if now_level < protection_level:
            matrix[now_level][0] = 1 - success_rate
        else:
            matrix[now_level][now_level - 1] = 1 - success_rate
    return matrix


if __name__ == '__main__':
    plt.figure(figsize=(8, 4.5), dpi=200)

    start = np.zeros(target_level + 1)
    start[0] = 1.0
    for protection_level in range(2, target_level + 1):
        matrix = stochastic_matrix(protection_level)
        distribution = start.copy()
        win_rate = np.zeros(work_times)
        expect = solve(protection_level)
        for i in range(work_times):
            # Let distribution = (target_level + 1)-dim vector
            # distribution[k] = possibility of having +k item after {i} actions
            distribution @= matrix
            win_rate[i] = distribution[target_level]
        # plt.plot(range(1, work_times + 1), win_rate,
        #          label=f'{protection_level}级保护' if protection_level < target_level else '不保护')
        # plt.text(0.5, 1.1, ha='center', va='center', transform=plt.gca().transAxes, fontsize=8,
        #          s=f'{base_player_level}级强化  +{enhancer_level} {enhancer_type}强化器  '
        #            f'{f"+{guzzling_pouch_level}暴饮之囊" if guzzling_pouch else ""}  {laboratory_level}级天文台  '
        #            f'茶={list(filter(lambda key: tea[key], tea.keys()))}\n'
        #            f'物品推荐等级={recommended_level}  目标强化等级={target_level}\n'
        #            f'图中曲线上高亮点表示期望强化次数', )
        plt.plot(range(1, work_times + 1), win_rate,
                 label=f'{protection_level} protect level' if protection_level < target_level else 'no protection')
        plt.text(0.5, 1.1, ha='center', va='center', transform=plt.gca().transAxes, fontsize=8,
                 s=f'{base_player_level} skill level | +{enhancer_level} {enhancer_type} enhancer | '
                   f'{"+" + str(guzzling_pouch_level) if guzzling_pouch else "No"} guzzling pouch | '
                   f'{laboratory_level} observatory level | tea={list(filter(lambda key: tea[key], tea.keys()))}\n'
                   f'item recommended level={recommended_level} | target level={target_level}\n'
                   f'points on the curve represent expected values', )
        plt.scatter(expect, win_rate[int(expect)])
    # plt.xlabel('强化次数')
    # plt.ylabel('成功率')
    plt.xlabel('actions')
    plt.ylabel('win rate')
    plt.legend()
    plt.show()
