import numpy as np
import torch

class ExtendedKalmanFilter(object):
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
        self.u = torch.tensor([[ux], [uy]])

        # Initial State
        self.x = torch.tensor([[0], [0], [0], [0]])

        # State Transition Matrix A
        self.A = torch.tensor([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        # Control Input Matrix B
        self.B = torch.tensor([[(self.dt ** 2) / 2, 0],
                           [0, (self.dt ** 2) / 2],
                           [self.dt, 0],
                           [0, self.dt]])

        # Measurement Mapping Matrix
        self.H = torch.tensor([[1, 0, 0, 0],
                           [0, 1, 0, 0]])

        # Process Noise Covariance
        self.Q = torch.tensor([[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
                           [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
                           [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
                           [0, (self.dt ** 3) / 2, 0,
                            self.dt ** 2]]) * std_a ** 2

        # Measurement Noise Covariance
        self.R = torch.tensor([[x_std ** 2, 0],
                           [0, y_std ** 2]])

        # Covariance Matrix
        self.P = torch.eye(4)

    def predict(self):

        # Update time state
        self.x = torch.mm(self.A, self.x) + torch.mm(self.B, self.u)

        # Calculate error covariance
        self.P = torch.mm(torch.mm(self.A, self.P), self.A.T) + self.Q
        return self.x[0]

    def update(self, z):

        x1 = self.x[0][0]
        y1 = self.x[1][0]
        x_sq = x1 * x1
        y_sq = y1 * y1
        den = x_sq + y_sq
        den1 = torch.sqrt(den)

        H = torch.tensor([[x1 / den1, y1 / den1, 0, 0], [y1 / den, -x1 / den, 0, 0]])

        S = torch.mm(H, torch.mm(self.P, H.T)) + self.R

        # Calculate the Kalman Gain
        K = torch.mm(torch.mm(self.P, H.T), torch.inverse(S))

        pred_x = self.x[0][0]
        pred_y = self.x[1][0]
        sumSquares = pred_x * pred_x + pred_y * pred_y
        pred_r = torch.sqrt(sumSquares)
        pred_b = torch.atan2(pred_x, pred_y) * 180 / np.pi
        y = torch.tensor([[pred_r], [pred_b]])

        res = z - y
        self.x = torch.round(self.x + torch.mm(K, res))

        # Update error covariance matrix
        self.P = torch.mm((torch.eye(H.shape[1]) - torch.mm(K, H)), self.P)
        return self.x[0]