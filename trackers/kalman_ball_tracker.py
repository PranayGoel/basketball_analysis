"""
A minimal, dependency-free 2D constant-velocity Kalman filter used to smooth and
predict basketball positions across frames.

Implemented by hand in pure numpy rather than pulling in `filterpy` -- the model
here is a textbook 4-state constant-velocity filter, simple enough to own directly
without taking on an unmaintained third-party dependency.
"""

import numpy as np


class ConstantVelocityKalmanFilter2D:
    """
    Minimal 2D constant-velocity Kalman filter tracking bbox CENTER position.

    State: [cx, cy, vx, vy]. Measurement: [cx, cy].
    """

    def __init__(self, process_noise=5.0, measurement_noise=10.0, initial_velocity_uncertainty=100.0):
        """
        Args:
            process_noise (float): Scalar multiplier for the process noise covariance Q
                (how much we expect the true constant-velocity model to be violated
                per frame, e.g. due to acceleration from a bounce or shot).
            measurement_noise (float): Scalar multiplier for the measurement noise
                covariance R (how much we trust a single detection's bbox center).
            initial_velocity_uncertainty (float): Variance seeded into the velocity
                terms of the covariance on `initialize()`, since the very first
                observation gives us a position but no velocity information.
        """
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_velocity_uncertainty = initial_velocity_uncertainty

        # State transition matrix: constant velocity, dt=1 frame.
        self.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)

        # Measurement matrix: we observe position (cx, cy) only.
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise covariance.
        self.Q = process_noise * np.eye(4)

        # Measurement noise covariance.
        self.R = measurement_noise * np.eye(2)

        self.x = np.zeros(4, dtype=float)
        self.P = np.eye(4, dtype=float)
        self._initialized = False

    @property
    def initialized(self):
        """bool: Whether `initialize()` has been called with a first observation."""
        return self._initialized

    def initialize(self, cx, cy):
        """
        Seed the filter's state with a first observation.

        Sets position to (cx, cy) and velocity to (0, 0). The covariance is set
        tight on position (we trust the first detection) and wide on velocity
        (we have no information about motion yet).

        Args:
            cx (float): Observed center x.
            cy (float): Observed center y.
        """
        self.x = np.array([cx, cy, 0.0, 0.0], dtype=float)
        self.P = np.diag([
            self.measurement_noise,
            self.measurement_noise,
            self.initial_velocity_uncertainty,
            self.initial_velocity_uncertainty,
        ]).astype(float)
        self._initialized = True

    def predict(self):
        """
        Advance the state estimate by one frame under the constant-velocity model.

        Returns:
            tuple[float, float]: Predicted (cx, cy) after the state transition.
        """
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return float(self.x[0]), float(self.x[1])

    def update(self, cx, cy):
        """
        Correct the state estimate using an observed measurement.

        Args:
            cx (float): Observed center x.
            cy (float): Observed center y.

        Returns:
            tuple[float, float]: Corrected (cx, cy) after the update step.
        """
        z = np.array([cx, cy], dtype=float)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        return float(self.x[0]), float(self.x[1])
