import numpy as np

class KalmanFilter(object):
    def __init__(self, dt, ux,uy, std_a, x_std, y_std):
        """
        :param dt: sampling time
        :param ux: acceleration in x-dir
        :param uy: acceleration in y-dir
        :param std_a: process noise magnitude
        :param x_std: sd in x-dir
        :param y_std: sd in y-dir
        """

        # Sampling time
        self.dt = dt

        # Control input variables
        self.u = np.array([[ux],[uy]])

        # Initial State
        self.x = np.array([[0], [0], [0], [0]])

        # State Transition Matrix A
        self.A = np.array([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])

        # Control Input Matrix B
        self.B = np.array([[(self.dt**2)/2, 0],
                            [0,(self.dt**2)/2],
                            [self.dt,0],
                            [0,self.dt]])

        # Measurement Mapping Matrix
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        # Process Noise Covariance
        self.Q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * std_a**2

        # Measurement Noise Covariance
        self.R = np.array([[x_std**2,0],
                           [0, y_std**2]])

        # Covariance Matrix
        self.P = np.eye(4)

    def predict(self):

        # Update time state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)

        # Calculate error covariance
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0]

    def update(self, z):

        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R

        # Calculate Kalman Gain
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))

        # Update error covariance matrix
        self.P = (np.eye(self.H.shape[1]) - (K * self.H)) * self.P
        return self.x[0]