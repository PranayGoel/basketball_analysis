import unittest

import numpy as np

from trackers.kalman_ball_tracker import ConstantVelocityKalmanFilter2D


class TestConstantVelocityKalmanFilter2D(unittest.TestCase):
    def test_predict_advances_position_by_velocity(self):
        kf = ConstantVelocityKalmanFilter2D()
        kf.initialize(0.0, 0.0)
        # Establish a clear rightward velocity: first predict+update moves the
        # state toward (10, 0), building up a positive x-velocity estimate.
        kf.predict()
        kf.update(10.0, 0.0)

        cx_before, _ = kf.x[0], kf.x[1]
        predicted_cx, predicted_cy = kf.predict()

        # With an established rightward velocity, predicting one more frame
        # ahead should move further in the positive-x direction.
        self.assertGreater(predicted_cx, cx_before)

    def test_update_corrects_toward_measurement(self):
        kf = ConstantVelocityKalmanFilter2D()
        kf.initialize(0.0, 0.0)
        kf.predict()

        state_before = np.array([kf.x[0], kf.x[1]])
        measurement = np.array([50.0, 50.0])
        distance_before = np.linalg.norm(state_before - measurement)

        kf.update(50.0, 50.0)

        state_after = np.array([kf.x[0], kf.x[1]])
        distance_after = np.linalg.norm(state_after - measurement)

        self.assertLess(distance_after, distance_before)

    def test_uninitialized_filter_flag(self):
        kf = ConstantVelocityKalmanFilter2D()
        self.assertFalse(kf.initialized)
        kf.initialize(1.0, 2.0)
        self.assertTrue(kf.initialized)

    def test_synthetic_parabolic_trajectory_tracking_error(self):
        # Deterministic, seeded synthetic trajectory simulating a shot/pass arc:
        # constant horizontal velocity + a parabolic (gravity-like) vertical component.
        rng = np.random.default_rng(42)
        n = 30
        t = np.arange(n, dtype=float)
        true_x = 5.0 * t
        true_y = 50.0 + 8.0 * t - 0.3 * (t ** 2)

        noise_std = 2.0
        obs_x = true_x + rng.normal(0, noise_std, n)
        obs_y = true_y + rng.normal(0, noise_std, n)

        kf = ConstantVelocityKalmanFilter2D(
            process_noise=5.0, measurement_noise=10.0, initial_velocity_uncertainty=100.0
        )
        kf.initialize(obs_x[0], obs_y[0])

        errors = []
        for i in range(1, n):
            predicted_cx, predicted_cy = kf.predict()
            true_next = np.array([true_x[i], true_y[i]])
            error = np.linalg.norm(np.array([predicted_cx, predicted_cy]) - true_next)
            errors.append(error)
            kf.update(obs_x[i], obs_y[i])

        mean_absolute_error = float(np.mean(errors))
        # Empirically ~3.0 with this seed/noise config; generous fixed threshold
        # keeps this a deterministic regression guard, not a flaky tolerance check.
        self.assertLess(mean_absolute_error, 6.0)


if __name__ == "__main__":
    unittest.main()
