import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv

# ------------------------
# Battery Model Parameters
# ------------------------
Q_battery = 5000  # Battery capacity in Coulombs (1.4 Ah)
V_nominal = 3.7   # Nominal battery voltage
R_internal = 0.015  # Internal resistance (Ohms)
dt = 1  # Time step (seconds)
time_steps = 1000  # Total simulation time

# Time array
time = np.arange(time_steps)

# Load current profile (Varying current load)
I_load = np.zeros(time_steps)
I_load[:300] = 0.5   # First 300 seconds
I_load[300:700] = 0.8  # Between 300s and 700s
I_load[700:] = 0.3    # After 700s

# ------------------------
# EKF Initialization
# ------------------------
n = 1  # State dimension (SoC)

# Initial SoC
SoC_actual = np.zeros(time_steps)
SoC_actual[0] = 0.9  # Assume 90% SoC at start
SoC_estimated = np.zeros(time_steps)
SoC_estimated[0] = 0.85  # Initial estimation with some error

# Process and Measurement Noise Covariances
Q = np.array([[1e-6]])  # Process noise covariance
R = np.array([[1e-3]])  # Measurement noise covariance

# Initial Covariance Matrix
P = np.array([[0.01]])  # Initial uncertainty

# For storing uncertainty bounds
upper_bound = np.zeros(time_steps)
lower_bound = np.zeros(time_steps)
upper_bound[0] = SoC_estimated[0] + 2 * np.sqrt(P[0,0])
lower_bound[0] = SoC_estimated[0] - 2 * np.sqrt(P[0,0])

# ------------------------
# EKF Functions
# ------------------------
def state_transition(SoC, I, dt):
    """ Battery SoC transition model """
    return np.clip(SoC - (I * dt) / Q_battery, 0, 1)

def measurement_function(SoC, I):
    """ Measurement model: Battery voltage """
    OCV = V_nominal * (0.8 + 0.2 * SoC)
    return OCV - (I * R_internal)

def state_jacobian(SoC, I, dt):
    """ Jacobian of state transition function """
    return np.array([[1.0]])  # Linear model

def measurement_jacobian(SoC, I):
    """ Jacobian of measurement function """
    return np.array([[0.2 * V_nominal]])  # Derivative of OCV w.r.t SoC

# ------------------------
# Main EKF Estimation Loop
# ------------------------
for t in range(1, time_steps):
    # Actual SoC evolution (ground truth)
    SoC_actual[t] = state_transition(SoC_actual[t-1], I_load[t-1], dt)
    
    # 1. Predict step
    # State prediction
    SoC_pred = state_transition(SoC_estimated[t-1], I_load[t-1], dt)
    
    # Covariance prediction
    F = state_jacobian(SoC_estimated[t-1], I_load[t-1], dt)
    P_pred = F @ P @ F.T + Q
    
    # 2. Update step
    # Measurement prediction
    z_pred = measurement_function(SoC_pred, I_load[t])
    
    # Linearization
    H = measurement_jacobian(SoC_pred, I_load[t])
    
    # Kalman gain calculation
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ inv(S)
    
    # Actual measurement with noise
    z_actual = measurement_function(SoC_actual[t], I_load[t]) + np.sqrt(R[0,0]) * np.random.randn()
    
    # State update
    SoC_estimated[t] = np.clip(SoC_pred + K * (z_actual - z_pred), 0, 1)
    
    # Covariance update
    P = (np.eye(n) - K @ H) @ P_pred
    
    # Ensure positive definiteness
    P = (P + P.T) * 0.5
    P = np.maximum(P, 1e-10 * np.eye(n))
    
    # Store uncertainty bounds
    upper_bound[t] = np.clip(SoC_estimated[t] + 2 * np.sqrt(P[0,0]), 0, 1)
    lower_bound[t] = np.clip(SoC_estimated[t] - 2 * np.sqrt(P[0,0]), 0, 1)

# ------------------------
# Plot Results (Identical to Original)
# ------------------------
plt.figure(figsize=(12, 6))
plt.plot(time/60, SoC_actual*100, label="Actual SoC", color='b')
plt.plot(time/60, SoC_estimated*100, label="Estimated SoC (EKF)", linestyle="dashed", color='r')
plt.fill_between(time/60, lower_bound*100, upper_bound*100, alpha=0.2, color='r', label='2σ Confidence')
plt.xlabel("Time (minutes)")
plt.ylabel("State of Charge (%)")
plt.legend(loc='best')
plt.grid(True)
plt.title("SoC Estimation using Extended Kalman Filter")

plt.figure(figsize=(12, 6))
plt.plot(time/60, np.abs(SoC_actual - SoC_estimated) * 100, color='g')
plt.xlabel("Time (minutes)")
plt.ylabel("SoC Estimation Error (%)")
plt.title("SoC Estimation Error Over Time")
plt.grid(True)

plt.figure(figsize=(12, 4))
plt.plot(time/60, I_load, 'k-', label="Current Profile")
plt.xlabel("Time (minutes)")
plt.ylabel("Current (A)")
plt.title("Battery Current Profile")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
