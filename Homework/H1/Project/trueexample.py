import numpy as np
import matplotlib.pyplot as plt
import copy

# 环境参数
n_states = 11  # 一共有11个状态
n_actions = 2  # 0: 左, 1: 右
gamma = 0.99 # 折扣因子
alpha = 0.99  # 优势学习算子中的学习率

# 设置随机种子，确保结果可重复
np.random.seed(0)

# 奖励函数
def reward(state):
    """到达状态 state 时获得的奖励。"""
    if state == 0:
        return 3.0
    elif 1 <= state <= 4:
        return -1.0
    elif state == 5:
        return 0.0
    elif 6 <= state <= 10:
        return 1.0
    else:
        raise ValueError("Invalid state")


# 转移函数
def transition(state, action):
    # action: 0=left, 1=right
    # 如果向左走
    if action == 0:  # left
        # intended: 期望状态,opposite: 相反状态
        intended = state - 1
        opposite = state + 1
    # 如果向右走
    else:  # right
        intended = state + 1
        opposite = state - 1
    
    # 处理边界情况
    intended = max(0, min(n_states-1, intended))
    opposite = max(0, min(n_states-1, opposite))
    
    # return [(0.7, intended), (0.3, opposite)]
    return [(intended, 0.7), (opposite, 0.3)]


# 计算真实最优价值函数 V^*
def compute_optimal_V(gamma=0.99, tol=1e-12):
    """
    对"永远向左"这一固定策略做策略评估，返回其状态价值 V^*(s)。
    在题目中已给出该策略就是最优策略。
    """
    V = np.zeros(n_states)
    while True:
        V_old = V.copy()
        for s in range(n_states):
            # 执行"向左"动作后，根据transition(s,0)计算下一步期望
            val = 0.0
            for (s_next, prob) in transition(s, 0):
                r = reward(s_next)
                val += prob * (r + gamma * V_old[s_next])
            V[s] = val
        # 判断收敛
        if np.max(np.abs(V - V_old)) < tol:
            break
    return V



# 计算真实最优V值
optimal_V = compute_optimal_V(gamma=gamma)

# Bellman最优算子
def bellman_optimal_operator(Q):
    """
    Bellman 最优算子: 给定当前 Q，返回对所有 Q(s,a) 的更新结果。
    """
    new_Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in [0, 1]:
            total = 0.0
            for (s_next, prob) in transition(s, a):
                r = reward(s_next)
                total += prob * (r + gamma * np.max(Q[s_next]))
            new_Q[s, a] = total
    return new_Q

# 优势学习算子
def advantage_learning_operator(Q):
    """
    优势学习更新: 给定当前 Q，返回对所有 Q(s,a) 的更新结果。
    按照公式: T_AL Q(s,a) = r(s,a) + α(Q(s,a) - max_a Q(s,a)) + γE[max_a' Q(s',a')]
    """
    new_Q = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a in [0,1]:
            # 计算 r(s,a) 项
            r_sa = 0.0
            for (s_next, prob) in transition(s, a):
                r_sa += prob * reward(s_next)
            
            # 计算 γE[max_a' Q(s',a')] 项
            expected_future = 0.0
            for (s_next, prob) in transition(s, a):
                expected_future += prob * np.max(Q[s_next])
            expected_future *= gamma
            
            # 计算 α(Q(s,a) - max_a Q(s,a)) 项
            advantage_term = alpha * (Q[s, a] - np.max(Q[s]))
            
            # 组合所有项
            new_Q[s, a] = r_sa + advantage_term + expected_future
    return new_Q

def evaluate_policy(policy, gamma=0.99, tol=1e-8, max_iter=1000):
    V = np.zeros(n_states)
    for _ in range(max_iter):
        V_old = V.copy()
        for s in range(n_states):
            a = policy[s]
            val = 0.0
            for (s_next, prob) in transition(s, a):
                r = reward(s_next)
                val += prob * (r + gamma * V_old[s_next])
            V[s] = val
        if np.max(np.abs(V - V_old)) < tol:
            break
    return V


# 计算性能界 ||V* - V^{π_k}||∞
def compute_performance_bound(optimal_V, current_V):
    """计算与 V^* 的无穷范数差"""
    return np.max(np.abs(optimal_V - current_V))

# 计算动作间隔
def compute_action_gap(Q):
    """计算动作间隔 (Action Gap): 每个状态下左右动作的Q值差异，然后取平均"""
    gaps = np.zeros(n_states)
    for s in range(n_states):
        gaps[s] = Q[s, 0] - Q[s, 1]  # 左(0) - 右(1)
    return np.mean(gaps)

# 主实验函数
def run_experiment(operator, n_iterations=400):
    # 初始化Q值
    np.random.seed(0)
    Q = 10 * np.random.rand(n_states, n_actions)
    
    performance_bounds = []
    action_gaps = []
    
    for _ in range(n_iterations):
        # 应用算子更新Q值
        Q = operator(Q)
        
        # 获取当前贪婪策略
        current_policy = np.argmax(Q, axis=1)
        
        # 评估当前策略的真实价值函数
        current_V = evaluate_policy(current_policy)
        
        # 计算性能界
        bound = compute_performance_bound(optimal_V, current_V)
        performance_bounds.append(bound)
        
        # 计算动作间隔
        gap = compute_action_gap(Q)
        action_gaps.append(gap)
    

    return performance_bounds, action_gaps, Q

# 运行实验
n_iterations = 400
np.random.seed(0)  # 重置随机种子，确保两次实验使用相同的初始Q值
bellman_bounds, bellman_gaps, Q_bellman = run_experiment(bellman_optimal_operator, n_iterations)
np.random.seed(0)  # 重置随机种子，确保两次实验使用相同的初始Q值
al_bounds, al_gaps, Q_adv = run_experiment(advantage_learning_operator, n_iterations)

gap_diff = []
for i in range(n_iterations):
    gap_diff.append(al_gaps[i] - bellman_gaps[i])
print(f"gap的差值:{gap_diff}")

fig, axs = plt.subplots(1, 2, figsize=(14, 5), sharey=False, sharex=True)

iters = np.arange(n_iterations)

# 左图: 性能界 ||V^* - V^π||∞
axs[0].plot(iters, bellman_bounds, label="Bellman Optimal", lw=2)
axs[0].plot(iters, al_bounds, label="Advantage Learning", lw=2)
axs[0].set_xlabel("Iterations")
axs[0].set_ylabel(r"Performance Bound ||V* - V^{π_k}||∞")
axs[0].set_title("Performance Bound over Iterations")
axs[0].grid(True)
axs[0].legend()

# 右图: 动作间隔 (Action Gap)
axs[1].semilogy(iters, np.abs(bellman_gaps), label="Bellman optimal")  # 使用对数坐标
axs[1].semilogy(iters, np.abs(al_gaps), label="Advantage Learning")  # 使用对数坐标
axs[1].set_xlabel("Iterations")
axs[1].set_ylabel("Action Gap")
axs[1].set_title("Action Gap over Iterations")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()


