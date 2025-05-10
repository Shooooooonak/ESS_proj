import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import cholesky


def run_srckf(window_size, plot=False):
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
    I_load[:300] = 0.5
    I_load[300:700] = 0.8
    I_load[700:] = 0.3

    n = 1  # State dimension (SoC)

    # Initial SoC
    SoC_actual = np.zeros(time_steps)
    SoC_actual[0] = 0.9
    SoC_estimated = np.zeros(time_steps)
    SoC_estimated[0] = 0.85

    # Process and Measurement Noise Covariances
    Q = np.array([[1e-6]])
    R = np.array([[1e-3]])

    # Initial Covariance Matrix
    P = np.array([[0.01]])
    sqrtP = cholesky(P, lower=True)

    # For storing uncertainty bounds
    upper_bound = np.zeros(time_steps)
    lower_bound = np.zeros(time_steps)
    upper_bound[0] = SoC_estimated[0] + 2 * np.sqrt(P[0,0])
    lower_bound[0] = SoC_estimated[0] - 2 * np.sqrt(P[0,0])

    # Cubature points
    def generate_cubature_points(n):
        points = np.zeros((2*n, n))
        for i in range(n):
            points[i, i] = np.sqrt(n)
            points[i+n, i] = -np.sqrt(n)
        return points
    cubature_points = generate_cubature_points(n)

    # State and measurement functions
    def state_transition(SoC, I, dt):
        return np.clip(SoC - (I * dt) / Q_battery, 0, 1)

    def measurement_function(SoC, I):
        OCV = V_nominal * (0.8 + 0.2 * SoC)
        V_terminal = OCV - (I * R_internal)
        return V_terminal

    # Buffer for residuals (for adaptive noise estimation)
    error_buffer = []

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
        P = P_pred - K @ P_xz.T
        P = (P + P.T) / 2
        P = np.maximum(P, 1e-10 * np.eye(n))
        sqrtP = cholesky(P, lower=True)

        # Store uncertainty bounds (2-sigma)
        upper_bound[t] = np.clip(SoC_estimated[t] + 2 * np.sqrt(P[0,0]), 0, 1)
        lower_bound[t] = np.clip(SoC_estimated[t] - 2 * np.sqrt(P[0,0]), 0, 1)

        # --- Adaptive noise update using moving window ---
        # Compute residual (innovation)
        residual = SoC_actual[t] - SoC_estimated[t]
        error_buffer.append(residual)
        if len(error_buffer) > window_size:
            error_buffer.pop(0)
        if len(error_buffer) == window_size:
            # Eq. 18/19 in [1]: H_k = (1/M) sum_{i=k-M+1}^{k} e_i^2
            Hk = np.mean(np.square(error_buffer))
            # Q_k = K_k * H_k * K_k^T
            Q = np.array([[float(K[0,0] * Hk * K[0,0])]])
            # R_k = H_k - C_k * P_k^- * C_k^T
            # For this model, C_k = 1, P_k^- = P_pred
            R_candidate = float(Hk - P_pred[0,0])
            R = np.array([[max(R_candidate, 1e-6)]])  # regularize to avoid negative/zero
            Q = np.clip(Q, 1e-8, 1)
            R = np.clip(R, 1e-6, 1)

    # Compute RMSE for fitness
    rmse = np.sqrt(np.mean((SoC_actual - SoC_estimated) ** 2))

    if plot:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        plt.plot(time/60, SoC_actual*100, label="Actual SoC", color='b')
        plt.plot(time/60, SoC_estimated*100, label="Estimated SoC (SRCKF)", linestyle="dashed", color='r')
        plt.fill_between(time/60, lower_bound*100, upper_bound*100, alpha=0.2, color='r', label='2σ Confidence')
        plt.xlabel("Time (minutes)")
        plt.ylabel("State of Charge (%)")
        plt.legend(loc='best')
        plt.grid(True)
        plt.title("SoC Estimation using PSO Adaptive SRCKF")
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

    return rmse  # or return SoC_actual, SoC_estimated if you prefer

def pso_optimize_srckf():
    num_particles = 20
    num_iterations = 20
    w = 0.7
    c1 = 2.0
    c2 = 2.0
    min_window = 5
    max_window = 100

    positions = np.random.uniform(min_window, max_window, num_particles)
    velocities = np.zeros(num_particles)
    personal_best_positions = np.copy(positions)
    personal_best_scores = np.array([run_srckf(pos) for pos in positions])
    global_best_idx = np.argmin(personal_best_scores)
    global_best_position = personal_best_positions[global_best_idx]

    for iter in range(num_iterations):
        for i in range(num_particles):
            r1, r2 = np.random.rand(), np.random.rand()
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - positions[i]) +
                             c2 * r2 * (global_best_position - positions[i]))
            positions[i] += velocities[i]
            positions[i] = np.clip(positions[i], min_window, max_window)
            score = run_srckf(positions[i])
            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_positions[i] = positions[i]
        global_best_idx = np.argmin(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx]
        print(f"Iteration {iter+1}, Best RMSE: {personal_best_scores[global_best_idx]:.6f}, Window: {global_best_position}")

    return int(global_best_position)

if __name__ == "__main__":
    optimal_window = pso_optimize_srckf()
    print(f"Optimal window size found by PSO: {optimal_window}")
    run_srckf(optimal_window, plot = True)