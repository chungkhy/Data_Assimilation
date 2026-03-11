import numpy as np

class KalmanFilter:
    """
    Classical discrete-time Kalman Filter for linear Gaussian systems:

        x_k = F x_{k-1} + B u_k + w_k,     w_k ~ N(0, Q)
        y_k = H x_k + v_k,                v_k ~ N(0, R)

    Attributes are NumPy arrays with standard shapes:
      x: (n, 1) state mean
      P: (n, n) state covariance
      F: (n, n) state transition
      B: (n, m) control matrix (optional)
      H: (p, n) observation matrix
      Q: (n, n) process noise covariance
      R: (p, p) observation noise covariance
    """

    def __init__(self, x0, P0, F, H, Q, R, B=None):
        self.x = self._col(x0)
        self.P = np.array(P0, dtype=float)
        self.F = np.array(F, dtype=float)
        self.H = np.array(H, dtype=float)
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)
        self.B = None if B is None else np.array(B, dtype=float)

        self._validate_shapes()

    @staticmethod
    def _col(x):
        x = np.array(x, dtype=float)
        return x.reshape(-1, 1)

    def _validate_shapes(self):
        n = self.x.shape[0]
        if self.P.shape != (n, n):
            raise ValueError(f"P must be {(n, n)}, got {self.P.shape}")
        if self.F.shape != (n, n):
            raise ValueError(f"F must be {(n, n)}, got {self.F.shape}")
        if self.Q.shape != (n, n):
            raise ValueError(f"Q must be {(n, n)}, got {self.Q.shape}")

        p = self.H.shape[0]
        if self.H.shape[1] != n:
            raise ValueError(f"H must be (p, n) with n={n}, got {self.H.shape}")
        if self.R.shape != (p, p):
            raise ValueError(f"R must be {(p, p)}, got {self.R.shape}")

        if self.B is not None:
            m = self.B.shape[1]
            if self.B.shape[0] != n:
                raise ValueError(f"B must be (n, m) with n={n}, got {self.B.shape}")
            # u will be checked at runtime

    def predict(self, u=None):
        """Time update (forecast): returns (x_pred, P_pred)."""
        if u is None or self.B is None:
            self.x = self.F @ self.x
        else:
            u = self._col(u)
            m = self.B.shape[1]
            if u.shape != (m, 1):
                raise ValueError(f"u must be {(m, 1)}, got {u.shape}")
            self.x = self.F @ self.x + self.B @ u

        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy(), self.P.copy()

    def update(self, y):
        """Measurement update (analysis): returns (x_upd, P_upd, K, innovation)."""
        y = self._col(y)
        p = self.H.shape[0]
        if y.shape != (p, 1):
            raise ValueError(f"y must be {(p, 1)}, got {y.shape}")

        # Innovation
        innovation = y - (self.H @ self.x)

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain (use solve for numerical stability vs inverse)
        K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(p))

        # State update
        self.x = self.x + K @ innovation

        # Joseph stabilized covariance update (more numerically stable)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ self.P @ (I - K @ self.H).T + K @ self.R @ K.T

        return self.x.copy(), self.P.copy(), K, innovation


if __name__ == "__main__":
    # Example: 1D constant-velocity model
    # State x = [position, velocity]^T
    dt = 1.0

    F = np.array([[1.0, dt],
                  [0.0, 1.0]])

    # Observe position only: y = position + noise
    H = np.array([[1.0, 0.0]])

    # Process noise: assume random acceleration
    sigma_a = 0.5  # accel std
    Q = (sigma_a**2) * np.array([[dt**4/4, dt**3/2],
                                 [dt**3/2, dt**2]])

    sigma_y = 2.0  # measurement std
    R = np.array([[sigma_y**2]])

    x0 = np.array([0.0, 1.0])      # start at 0, moving at 1 unit/s
    P0 = np.diag([10.0, 10.0])     # high initial uncertainty

    kf = KalmanFilter(x0=x0, P0=P0, F=F, H=H, Q=Q, R=R)

    # Simulated measurements (positions)
    measurements = [0.9, 2.2, 2.8, 4.6, 5.1, 7.4, 7.9, 9.8]

    print("k  meas    pos_est  vel_est")
    for k, z in enumerate(measurements):
        kf.predict()
        x, P, K, innov = kf.update([z])
        pos_est, vel_est = float(x[0]), float(x[1])
        print(f"{k:1d}  {z:5.2f}   {pos_est:7.3f}  {vel_est:7.3f}")