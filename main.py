import numpy as np
from matplotlib import pyplot as plt

from constant import *
from settings import *

# np.set_printoptions(linewidth=1000, precision=4, suppress=True)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def expectation(protect_level: int, st: Setting):
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
    coefficient = np.zeros((st.target_level + 1, st.target_level + 1))
    for now_level in range(st.target_level + 1):
        success_rate = st.enhance_success_rate(now_level)
        if now_level == st.target_level:
            coefficient[now_level][now_level] = 1.0
            continue
        coefficient[now_level][now_level] = 1.0
        if now_level < protect_level:
            coefficient[now_level][0] += success_rate - 1
        else:
            coefficient[now_level][now_level - 1] = success_rate - 1
        if now_level + 1 < st.target_level:
            coefficient[now_level][now_level + 1] = -success_rate * (1 - st.blessed_rate)
            coefficient[now_level][now_level + 2] = -success_rate * st.blessed_rate
        else:
            coefficient[now_level][now_level + 1] = -success_rate
    return np.linalg.solve(coefficient, np.ones(st.target_level + 1))[0]


def stochastic_matrix(protect_level: int, st: Setting):
    """
    result[i][j] = when you have a +i item, result[i][j] possibility to get a +j item after one action
    """
    result = np.zeros((st.target_level + 1, st.target_level + 1))
    for now_level in range(st.target_level + 1):
        success_rate = st.enhance_success_rate(now_level)
        if now_level == st.target_level:
            result[now_level][now_level] = 1.0
            continue
        # success
        if now_level + 1 < st.target_level:
            result[now_level][now_level + 1] = success_rate * (1 - st.blessed_rate)
            result[now_level][now_level + 2] = success_rate * st.blessed_rate
        else:
            result[now_level][now_level + 1] = success_rate
        # fail
        if now_level < protect_level:
            result[now_level][0] = 1 - success_rate
        else:
            result[now_level][now_level - 1] = 1 - success_rate
    return result




if __name__ == '__main__':
    st = Setting()
    plt.figure(figsize=(8, 4.5), dpi=100)

    start = np.zeros(st.target_level + 1)
    start[0] = 1.0
    for protect_level in range(2, st.target_level + 1):
        stochastic = stochastic_matrix(protect_level, st)
        # print(np.linalg.cond(stochastic))
        matrix = stochastic.copy()
        win_rate = np.zeros(st.enhance_times)
        for i in range(st.enhance_times):
            # win_rate[k] = x A^k = (0, 0, 0, ..., 0, 0, 1) (stochastic)^k = sum of last column of (stochastic)^k
            matrix @= stochastic
            win_rate[i] = np.sum(matrix[:, st.target_level])
        plt.plot(range(1, st.enhance_times + 1), win_rate,
                 label=f'{protect_level}级保护' if protect_level < st.target_level else '不保护')
        plt.text(0.5, 1.1, ha='center', va='center', transform=plt.gca().transAxes, fontsize=8,s='')
        expect = expectation(protect_level,st)
        plt.scatter(expect, win_rate[int(expect)])
    plt.xlabel('强化次数')
    plt.ylabel('成功率')
    plt.legend()
    plt.show()
