import numpy as np
import matplotlib.pyplot as plt

# Constants
mass = 1470     # Mass of the car (kg)
I_z = 1900      # Moment of inertia about the z-axis (kg*m^2)
l_f = 1.04      # Distance from the center of mass to the front axle (m)
l_r = 1.56      # Distance from the center of mass to the rear axle (m)
C_f = 71000     # Front tire cornering stiffness (N/rad)
C_r = 47000     # Rear tire cornering stiffness (N/rad)
V_x = 20        # Velocity of the car in the x-direction (m/s)

# Initial conditions
X_0 = 0           # Initial x-position (m)
Y_0 = 0           # Initial y-position (m)
Psi_0 = 0         # Initial yaw angle (rad)
x_dot0 = V_x      # Initial longitudinal velocity (m/s)
y_dot0 = 0        # Initial lateral velocity (m/s)
psi_dot0 = 0      # Initial yaw rate (rad/s)

# Time parameters
dt = 0.01       # Time step (s)
t_end = 10      # End time (s)
timesteps = int(t_end / dt)  # Number of time steps

# delta calculation parameters
delta_amplitude = 0.10016  # radians
delta_frequency = 0.2      # Hz

# non linear Dynamics function
def dynamics(state, delta):
    x_dot, y_dot, psi_dot = state

    a_f = delta - np.arctan2((y_dot + l_f * psi_dot), x_dot)
    a_r = -np.arctan2((y_dot - l_r * psi_dot), x_dot)

    F_yf = C_f * a_f
    F_yr = C_r * a_r

    x_ddot = 0
    y_ddot = (F_yf * np.cos(delta) + F_yr) / mass - psi_dot * x_dot
    psi_ddot = (l_f * F_yf - l_r * F_yr) / I_z

    return np.array([x_ddot, y_ddot, psi_ddot])

# State-space dynamics function
def state_space_dynamics(state, delta):
    y_dot, psi_dot = state
    A = np.array([
        [-(C_f + C_r) / (mass * V_x), (C_f * l_f - C_r * l_r) / (mass * V_x) - V_x],
        [(C_f * l_f - C_r * l_r) / (I_z * V_x), -(C_f * l_f**2 + C_r * l_r**2) / (I_z * V_x)]
    ])
    B = np.array([
        [C_f / mass],
        [C_f * l_f / I_z]
    ]) * delta
    ss_dot = (A.dot(np.array([y_dot, psi_dot])) + B).flatten()
    return np.array([ss_dot[0],ss_dot[1]])

# Trajectory calculation function
def trajectory_cal(state, traj):
    x_dot, y_dot, psi_dot = state
    X, Y, Psi = traj

    X_dot = x_dot * np.cos(Psi) - y_dot * np.sin(Psi)
    Y_dot = x_dot * np.sin(Psi) + y_dot * np.cos(Psi)
    Psi_dot = psi_dot

    return np.array([X_dot, Y_dot, Psi_dot])

# starting state initialization
stateVector = np.array([x_dot0, y_dot0, psi_dot0])
ss_stateVector = np.array([y_dot0, psi_dot0])
trajVector = np.array([X_0, Y_0, Psi_0])

# Arrays for storing results
stateArray = np.zeros((timesteps, 3))
trajectory_nl = np.zeros((timesteps, 3))
trajectory_nl[0] = trajVector
acc_y_array = np.zeros(timesteps)
psi_dot_array = np.zeros(timesteps)
ss_acc_y_array = np.zeros(timesteps)
ss_psi_dot_array = np.zeros(timesteps)
time_array = np.linspace(0, t_end, timesteps)

# Main loop
for i in range(1, timesteps):
    delta = delta_amplitude * np.sin(2 * np.pi * delta_frequency * time_array[i])

    # Dynamics calculations
    derivatives = dynamics(stateVector, delta)
    stateVector = stateVector + derivatives * dt
    stateArray[i] = stateVector
    acc_y_array[i] = derivatives[1]
    psi_dot_array[i] = derivatives[2]

    # Trajectory update
    traj_derivatives = trajectory_cal(stateVector, trajVector)
    trajVector = trajVector + traj_derivatives * dt
    trajectory_nl[i] = trajVector

    # State-space dynamics calculations
    ss_derivatives = state_space_dynamics(ss_stateVector, delta)
    ss_stateVector = ss_stateVector + (ss_derivatives * dt).flatten()
    ss_acc_y_array[i] = ss_derivatives[0]
    ss_psi_dot_array[i] = ss_derivatives[1]



# Trajectory plot
plt.figure(figsize=(10, 5))
plt.plot(trajectory_nl[:, 0], trajectory_nl[:, 1], label='Nonlinear Dynamics Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Vehicle Trajectory')
plt.legend()
plt.grid(True)

# Lateral Acceleration and Yaw Rate comparison plots
plt.figure(figsize=(14, 6))

# Lateral Acceleration plot
plt.subplot(1, 2, 1)
plt.plot(time_array, acc_y_array, label='Nonlinear Dynamics acc_y')
plt.plot(time_array, ss_acc_y_array, '--', label='State-Space Dynamics acc_y')
plt.xlabel('Time (s)')
plt.ylabel('Lateral Acceleration (m/s^2)')
plt.title('Lateral Acceleration Comparison')
plt.legend()
plt.grid(True)

# Yaw Rate plot
plt.subplot(1, 2, 2)
plt.plot(time_array, psi_dot_array, label='Nonlinear Dynamics ψ_dot')
plt.plot(time_array, ss_psi_dot_array, '--', label='State-Space Dynamics ψ_dot')
plt.xlabel('Time (s)')
plt.ylabel('Yaw Rate (rad/s)')
plt.title('Yaw Rate Comparison')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
