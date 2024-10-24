import numpy as np
import matplotlib.pyplot as plt

def get_storm_prob(x, x_eye, sigma):
    return np.exp(-np.linalg.norm(np.array(x) - np.array(x_eye))**2 / (2 * sigma**2))

# indicator function
def get_reward(x, x_goal):
    if x == x_goal:
        return 1
    return 0

def get_action_probs(x, x_eye, sigma, action):
    w = get_storm_prob(x, x_eye, sigma)
    action_probs = {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}
    for a in action_probs.keys():
        if a == action:
            action_probs[a] = 1 - w
        else:
            action_probs[a] = w / 3
    return action_probs

def get_next_state(x, action, n):
    i, j = x
    if action == "up":
        return (max(i-1, 0), j)
    if action == "down":
        return (min(i+1, n-1), j)
    if action == "left":
        return (i, max(j-1, 0))
    if action == "right":
        return (i, min(j+1, n-1))

def bellman_update(V, x, x_eye, x_goal, sigma, gamma, n):
    if x == x_goal: # terminal state
        return 1
    max_val = -np.inf
    for action in ["up", "down", "left", "right"]:
        action_probs = get_action_probs(x, x_eye, sigma, action)
        val = 0
        for a, p in action_probs.items():
            x_next = get_next_state(x, a, n)
            val += p * (get_reward(x_next, x_goal) + gamma * V[x_next])
        max_val = max(max_val, val)
    return max_val

def run_value_iteration(V_0, x_eye, x_goal, sigma, gamma, n, eps = 1e-6):
    V = V_0.copy()
    while True:
        V_new = V.copy()
        for i in range(n):
            for j in range(n):
                x = (i, j)
                V_new[x] = bellman_update(V, x, x_eye, x_goal, sigma, gamma, n)
        if np.linalg.norm(V_new - V) < eps:
            break
        V = V_new
    return V

def get_optimal_policy(V_star, x_eye, x_goal, sigma, gamma, n):
    policy = np.zeros((n, n), dtype=f'<U10')
    for i in range(n):
        for j in range(n):
            x = (i, j)
            max_val = -np.inf
            for action in ["up", "down", "left", "right"]:
                action_probs = get_action_probs(x, x_eye, sigma, action)
                val = 0
                for a, p in action_probs.items():
                    x_next = get_next_state(x, a, n)
                    val += p * (get_reward(x_next, x_goal) + gamma * V_star[x_next])
                if val > max_val:
                    max_val = val

                    policy[x[0], x[1]] = action
    return policy

def simulate_trajectory(x_0, pi_star, x_eye, x_goal, sigma, gamma, n, N):
    x = x_0
    trajectory = [x]
    for _ in range(N): # run the MDP for N steps
        action = pi_star[x] # ideal action 
        action_probs = get_action_probs(x, x_eye, sigma, action)
        action = np.random.choice(["up", "down", "left", "right"], p=list(action_probs.values())) # get the next action based on the storm probabilities
        x = get_next_state(x, action, n) # move to the next state
        trajectory.append(x)
    return trajectory

def convert_policy_to_numeric(pi_star, n=20):
    pi_star_numeric = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if pi_star[i, j] == "up":
                pi_star_numeric[i, j] = 0
            elif pi_star[i, j] == "down":
                pi_star_numeric[i, j] = 1
            elif pi_star[i, j] == "left":
                pi_star_numeric[i, j] = 2
            elif pi_star[i, j] == "right":
                pi_star_numeric[i, j] = 3
    return pi_star_numeric

def main():
    # Constants
    n = 20
    sigma = 10
    gamma = 0.95
    x_eye = (15, 15)
    x_goal = (19, 9)

    V_0 = np.zeros((n, n))

    V_star = run_value_iteration(V_0, x_eye, x_goal, sigma, gamma, n)

    # Plot heatmap, part a
    plt.imshow(V_star.T, cmap='hot', origin='lower')
    plt.colorbar()
    plt.title('Optimal Value Function')
    plt.xlabel('x1')
    plt.ylabel('x2')

    pi_star = get_optimal_policy(V_star, x_eye, x_goal, sigma, gamma, n)
    pi_star_numeric = convert_policy_to_numeric(pi_star, n)
    
    N = 100
    x_init = (0, 19)
    trajectory = simulate_trajectory(x_init, pi_star, x_eye, x_goal, sigma, gamma, n, N)

    # Plot trajectory, part b
    plt.figure()
    plt.imshow(pi_star_numeric.T, cmap='hot', origin='lower')
    plt.plot([x[0] for x in trajectory], [x[1] for x in trajectory], 'b-')
    plt.colorbar()
    plt.title('Optimal Policy with Trajectory')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(['Trajectory'])
    plt.show()


if __name__ == "__main__":
    main()