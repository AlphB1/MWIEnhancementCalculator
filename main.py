import numpy as np
from matplotlib import pyplot as plt
from settings import *

np.set_printoptions(linewidth=1000, precision=5, suppress=True)
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


def protect_expectation(protect_level: int, st: Setting):
    pass


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


def get_distribution(protect_level: int, protect_times: int, enhance_times: int, st: Setting, tolerance=1e-8):
    """
    Let distribution = protection_level × (protect_times+1) matrix
    After k actions, distribution[i][j] = possibility of owning +i item and having used j protection
    """
    if protect_times == np.inf:
        return np.linalg.matrix_power(stochastic_matrix(protect_level, st), enhance_times)[0][st.target_level]
    assert 2 <= protect_level <= st.target_level
    accomplish_rate = np.zeros((enhance_times + 1, protect_times + 1))
    no_protect_matrix = np.zeros((st.target_level + 1, st.target_level + 1))
    protect_vector = np.zeros(st.target_level - protect_level)
    for now_level in range(st.target_level + 1):
        if now_level == st.target_level:
            no_protect_matrix[now_level][now_level] = 1.0
            continue
        success_rate = st.enhance_success_rate(now_level)
        # success
        if now_level + 1 < st.target_level:
            no_protect_matrix[now_level][now_level + 1] = success_rate * (1 - st.blessed_rate)
            no_protect_matrix[now_level][now_level + 2] = success_rate * st.blessed_rate
        else:
            no_protect_matrix[now_level][now_level + 1] = success_rate
        # fail
        if now_level < protect_level:
            no_protect_matrix[now_level][0] = 1 - success_rate
        else:
            protect_vector[now_level - protect_level] = 1 - success_rate
    distribution = np.zeros((protect_times + 1, st.target_level + 1))
    distribution[0][0] = 1.0
    for i in range(1, enhance_times + 1):
        if i % 100 == 0:
            print(i)
        next = np.vstack((distribution[:-1, ] @ no_protect_matrix, distribution[-1]))
        protect_slice = distribution[:-1, protect_level:st.target_level]
        next[1:, protect_level - 1:st.target_level - 1] += protect_slice * protect_vector
        distribution = next
        for j in range(1, protect_times + 1):
            accomplish_rate[i][j] = accomplish_rate[i][j - 1] + distribution[j - 1][-1]
    assert abs(1 - np.sum(distribution)) < tolerance
    return distribution, accomplish_rate


def fixed_accomplish_rate(matrix, protect_cost, alpha=0.95):
    from bisect import bisect_left
    if matrix[-1][-1] < alpha:
        return None
    n, m = matrix.shape
    return min(((i, tmp if (tmp := bisect_left(matrix[i], alpha)) < m else np.inf) for i in range(n)),
               key=lambda x: (x[0] + protect_cost * x[1]))


if __name__ == '__main__':
    st = Setting()
    pl = 8
    pt = 50
    et = 30000
    accomplish_rate = get_distribution(pl, pt, et, st)[1]
    best_ratio = fixed_accomplish_rate(accomplish_rate, 1200)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=-120, roll=0)
    ax.set_xlim(0, et)
    ax.set_ylim(0, pt)
    ax.set_zlim(0.0, 1.0)
    ax.plot_surface(np.tile(np.arange(et + 1), (pt + 1, 1)).T,
                    np.tile(np.arange(pt + 1), (et + 1, 1)),
                    accomplish_rate, cmap='viridis',zorder=0)
    ax.set_xlabel('强化次数')
    ax.set_ylabel('准备垫子数量')
    ax.set_zlabel('出货概率')
    ax.scatter(best_ratio[0], best_ratio[1], accomplish_rate[best_ratio[0]][best_ratio[1]], color='red')
    print(best_ratio)
    plt.show()
    # plt.figure(figsize=(8, 4.5), dpi=100)
    #
    # start = np.zeros(st.target_level + 1)
    # start[0] = 1.0
    # for protect_level in range(2, st.target_level + 1):
    #     stochastic = stochastic_matrix(protect_level, st)
    #     # print(np.linalg.cond(stochastic))
    #     matrix = stochastic.copy()
    #     win_rate = np.zeros(st.enhance_times)
    #     for i in range(st.enhance_times):
    #         # win_rate[k] = x A^k = (0, 0, 0, ..., 0, 0, 1) (stochastic)^k = sum of last column of (stochastic)^k
    #         matrix @= stochastic
    #         win_rate[i] = np.sum(matrix[:, st.target_level])
    #     plt.plot(range(1, st.enhance_times + 1), win_rate,
    #              label=f'{protect_level}级保护' if protect_level < st.target_level else '不保护')
    #     plt.text(0.5, 1.1, ha='center', va='center', transform=plt.gca().transAxes, fontsize=8, s='')
    #     expect = expectation(protect_level, st)
    #     plt.scatter(expect, win_rate[int(expect)])
    # plt.xlabel('强化次数')
    # plt.ylabel('成功率')
    # plt.legend()
    # plt.show()
