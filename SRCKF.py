import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky

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
# I_load = np.piecewise(time,
#                       [time < 300, (time >= 300) & (time < 700), time >= 700],
#                      [0.5, 0.8, 0.3])  # Current in Amps

# Load current profile (Varying current load)
I_load = np.zeros(time_steps)  # Initialize with zeros

# Define different current phases
I_load[:300] = 0.5  # First 300 seconds
I_load[300:700] = 0.8  # Between 300s and 700s
I_load[700:] = 0.3  # After 700s


# ------------------------
# SRCKF Initialization
# ------------------------
n = 1  # State dimension (SoC)

# Initial SoC
SoC_actual = np.zeros(time_steps)
SoC_actual[0] = 0.9  # Assume 90% SoC at start
SoC_estimated = np.zeros(time_steps)
SoC_estimated[0] = 0.85  # Initial estimation with some error

# Process and Measurement Noise Covariances
Q = np.array([[1e-6]])  # Process noise covariance (reduced for better stability)
R = np.array([[1e-3]])  # Measurement noise covariance

# Initial Covariance Matrix
P = np.array([[0.01]])  # Initial uncertainty
sqrtP = cholesky(P, lower=True)  # Square Root of Covariance

# For storing uncertainty bounds
upper_bound = np.zeros(time_steps)
lower_bound = np.zeros(time_steps)
upper_bound[0] = SoC_estimated[0] + 2 * np.sqrt(P[0,0])
lower_bound[0] = SoC_estimated[0] - 2 * np.sqrt(P[0,0])

# ------------------------
# Cubature Kalman Filter Functions
# ------------------------

def state_transition(SoC, I, dt):
    """ Battery SoC transition model based on Coulomb Counting """
    return np.clip(SoC - (I * dt) / Q_battery, 0, 1)

def measurement_function(SoC, I):
    """ Measurement model: Simulated battery voltage with internal resistance effect """
    # Simplified OCV-SOC relationship
    OCV = V_nominal * (0.8 + 0.2 * SoC)  # More realistic voltage model
    V_terminal = OCV - (I * R_internal)
    return V_terminal

def generate_cubature_points(n):
    """ Generate the cubature points for n dimensions """
    points = np.zeros((2*n, n))
    for i in range(n):
        points[i, i] = np.sqrt(n)
        points[i+n, i] = -np.sqrt(n)
    return points

# Generate standard cubature points
cubature_points = generate_cubature_points(n)

# ------------------------
# Main SRCKF Estimation Loop
# ------------------------
for t in range(1, time_steps):
    # Actual SoC evolution (ground truth)
    SoC_actual[t] = state_transition(SoC_actual[t-1], I_load[t-1], dt)
   
    # 1. Generate transformed sigma points
    X = np.zeros((2*n, n))
    for i in range(2*n):
        X[i] = SoC_estimated[t-1] + sqrtP @ cubature_points[i].reshape(n, 1)
   
    # 2. Time update (prediction)
    X_pred = np.zeros((2*n, n))
    for i in range(2*n):
        X_pred[i] = state_transition(X[i], I_load[t-1], dt)
   
    # 3. Compute predicted mean
    x_pred_mean = np.mean(X_pred, axis=0).reshape(n, 1)
   
    # 4. Compute predicted covariance
    X_centered = X_pred - x_pred_mean.T
    P_pred = (X_centered.T @ X_centered) / (2*n) + Q
   
    # 5. Square-root of predicted covariance
    sqrtP_pred = cholesky(P_pred, lower=True)
   
    # 6. Measurement update
    Z = np.zeros(2*n)
    for i in range(2*n):
        Z[i] = measurement_function(X_pred[i], I_load[t])
   
    # 7. Compute measurement mean
    z_mean = np.mean(Z)
   
    # 8. Compute measurement covariance
    Z_centered = Z - z_mean
    P_zz = np.sum(Z_centered**2) / (2*n) + R[0,0]
   
    # 9. Compute cross-covariance
    P_xz = np.zeros((n, 1))
    for i in range(2*n):
        P_xz += (X_pred[i].reshape(n, 1) - x_pred_mean) * (Z[i] - z_mean) / (2*n)
   
    # 10. Compute Kalman gain
    K = P_xz / P_zz
   
    # 11. Update state estimate with actual measurement
    z_actual = measurement_function(SoC_actual[t], I_load[t]) + np.sqrt(R[0,0]) * np.random.randn()
    SoC_estimated[t] = np.clip(x_pred_mean + K * (z_actual - z_mean), 0, 1)
   
    # 12. Update covariance matrix
    P = P_pred - K * P_xz.T
   
    # Ensure P is positive definite
    P = (P + P.T) / 2
    P = np.maximum(P, 1e-10 * np.eye(n))
   
    # Get square root of P
    sqrtP = cholesky(P, lower=True)
   
    # Store uncertainty bounds (2-sigma)
    upper_bound[t] = np.clip(SoC_estimated[t] + 2 * np.sqrt(P[0,0]), 0, 1)
    lower_bound[t] = np.clip(SoC_estimated[t] - 2 * np.sqrt(P[0,0]), 0, 1)

# ------------------------
# Plot Results
# ------------------------
plt.figure(figsize=(12, 6))
plt.plot(time/60, SoC_actual*100, label="Actual SoC", color='b')
plt.plot(time/60, SoC_estimated*100, label="Estimated SoC (SRCKF)", linestyle="dashed", color='r')
plt.fill_between(time/60, lower_bound*100, upper_bound*100, alpha=0.2, color='r', label='2σ Confidence')
plt.xlabel("Time (minutes)")
plt.ylabel("State of Charge (%)")
plt.legend(loc='best')
plt.grid(True)
plt.title("SoC Estimation using Square Root CKF")

plt.figure(figsize=(12, 6))
plt.plot(time/60, np.abs(SoC_actual - SoC_estimated) * 100, color='g')
plt.xlabel("Time (minutes)")
plt.ylabel("SoC Estimation Error (%)")
plt.title("SoC Estimation Error Over Time")
plt.grid(True)

# Add current profile plot
plt.figure(figsize=(12, 4))
plt.plot(time/60, I_load, 'k-', label="Current Profile")
plt.xlabel("Time (minutes)")
plt.ylabel("Current (A)")
plt.title("Battery Current Profile")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()