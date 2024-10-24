import numpy as np
import cvxpy as cvx
import matplotlib.pyplot as plt

##### Part D #####
def generate_ellipsoid_points(M, num_points=100):
    L = np.linalg.cholesky(M)
    θ = np.linspace(0, 2*np.pi, num_points)
    u = np.column_stack([np.cos(θ), np.sin(θ)])
    x = u @ L.T
    return x

# Define the parameters
A = np.array([[0.9, 0.6], [0.0, 0.8]])
N = 4
B = np.array([[0], [1]])
Q = np.eye(2)
R = np.eye(1)
P = np.eye(2)
rx = 5
ru = 1
N = 4
x0 = np.array([0, -4.5])

M = cvx.Variable((2, 2), symmetric=True) # Ensure M is symmetric
objective = cvx.Maximize(cvx.log_det(M))
constraints = [
    M >> 0, 
    cvx.quad_form(A.T, M) - M << 0, 
    M << (rx**2) * np.eye(2)
]

prob = cvx.Problem(objective, constraints)
prob.solve()

status = prob.status
print(status)
M_opt = M.value
W_opt = np.linalg.inv(M_opt)
print(W_opt)

# Generate points for ellipsoids XT and AXT
XT_points = generate_ellipsoid_points(M_opt)
AXT_points = generate_ellipsoid_points(A @ M_opt)

# Generate points for X
r = np.sqrt(rx**2)
X_points = generate_ellipsoid_points(r**2 * np.eye(2))

# Plot the ellipsoids
fig, ax = plt.subplots()
ax.plot(XT_points[:, 0], XT_points[:, 1], label='XT')
ax.plot(AXT_points[:, 0], AXT_points[:, 1], label='AXT')
ax.plot(X_points[:, 0], X_points[:, 1], label='X')
ax.set_aspect('equal')
ax.legend()

##### Part E #####

##### Find Actual Trajectory #####
t_steps = 15
x = cvx.Variable((2, t_steps+1))
u = cvx.Variable((1, t_steps))
n, m = Q.shape[0], R.shape[0]

x_cvx = cvx.Variable((t_steps + 1, n))
u_cvx = cvx.Variable((t_steps, m))

cost = 0.0
constraints = [x_cvx[0] == x0]

for t in range(t_steps):
    cost += cvx.quad_form(x_cvx[t], Q) + cvx.quad_form(u_cvx[t], R)
    constraints += [x_cvx[t + 1] == A @ x_cvx[t] + B @ u_cvx[t]]
    constraints += [cvx.norm(x_cvx[t]) <= rx]
    constraints += [cvx.norm(u_cvx[t]) <= ru]

cost += cvx.quad_form(x_cvx[N], P)

# Solve the MPC problem
prob = cvx.Problem(cvx.Minimize(cost), constraints)
prob.solve()

# Extract the optimal values
x_opt = x_cvx.value
u_opt = u_cvx.value

# Plot the state trajectory
ax.plot(x_opt[:, 0], x_opt[:, 1], label='x_cvx', linewidth=3.0)
ax.set_aspect('equal')
ax.legend()

##### Find the Planned Trajectories #####
def solve_mpc(x0):
    x = cvx.Variable((2, N+1))
    u = cvx.Variable((1, N))
    cost = 0
    constraints = [x[:, 0] == x0]
    for t in range(N):
        cost += cvx.quad_form(x[:, t], Q) + cvx.quad_form(u[:, t], R)
        constraints += [x[:, t+1] == A @ x[:, t] + B @ u[:, t],
                        cvx.norm(x[:, t+1], 2) <= rx,
                        cvx.norm(u[:, t], 2) <= ru]
    cost += cvx.quad_form(x[:, N], P)
    problem = cvx.Problem(cvx.Minimize(cost), constraints)
    problem.solve()
    return x.value, u.value

x_traj = [x0]
u_traj = []

for t in range(t_steps):
    x0 = x_traj[-1]
    x_pred, u_pred = solve_mpc(x0)
    x_traj.append(x_pred[:, 1])
    u_traj.append(u_pred[:, 0])

x_traj = np.array(x_traj)
u_traj = np.array(u_traj)

# Overlay the planned trajectories at each time step
for t in range(t_steps):
    x0 = x_traj[t]
    x_pred, _ = solve_mpc(x0)
    ax.plot(x_pred[0, :], x_pred[1, :], 'b--')

plt.show()

# Plot the actual control trajectory
fig, ax = plt.subplots()
ax.plot(range(t_steps), u_opt, label='Actual Control Trajectory')
ax.set_xlabel('Time Step')
ax.set_ylabel('Control Input')
ax.legend()
plt.show()