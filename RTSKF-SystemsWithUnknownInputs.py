# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 01:59:10 2023

@author: Ghadeer SHAABAN
Ph.D. student at GIPSA-Lab
University Grenoble Alpes, France

This code provides a simple implementation of the Robust Two-Stage Kalman 
Filters (RTSKF) for Systems with Unknown Inputs. The code was developed from 
scratch due to the lack of existing straightforward implementations available
online. It serves as a valuable resource for those in search of a simple RTSKF
that may not be readily accessible elsewhere on the internet
(to the best of the author's knowledge).

Here I add a link to the original paper for the mathematics behind and the 
algorithm explaination: https://ieeexplore.ieee.org/document/895577
"""

# import Libraries
import numpy as np
import math
import matplotlib.pyplot as plt


"""
##############################################################################
###################       Dynamic and output Matrices      ###################      
##############################################################################
"""
nx = 3
ny = 2
nd = 1

A = np.array([[0.1, 0.5, 0.08], [0.6, 0.01, 0.04], [0.1, 0.7, 0.05]])

E = np.array([[0], [2], [1]])

H = np.array([[1, 1, 0], [0, 1, 1]])


"""
##############################################################################
###################        Generate True data              ###################      
##############################################################################
"""

N = 100  # number of samples
timeVector = np.linspace(0, N - 1, N)


# True input (which is unknown)
d_true = np.zeros((N, nd))
d_true[:, 0] = 50 * np.heaviside(timeVector, 1) - 100 * np.heaviside(
    timeVector - 20, 1
)


x0 = np.array([1, 1, 1])  # initial state
x_true_model = np.zeros((N, nx))  # the true model
x_true_model[0, :] = x0.reshape(-1)
for i in range(1, N):
    x_true_model[i, :] = A @ x_true_model[i - 1, :] + E @ d_true[i - 1, :]


"""
##############################################################################
##################            Generate noisy data           ##################      
##############################################################################
"""

sigma_x = np.sqrt(10)  # process noise std
sigma_y = np.sqrt(20)  # measurement noise std
Q = sigma_x**2 * np.eye(3)  # process noise covariance matrix
R = sigma_y**2 * np.eye(2)  # measurement noise covariance matrix

measurements_noise = np.random.normal(0, sigma_y, (N, ny))
process_noise = np.random.normal(0, sigma_x, (N, nx))


x_true = np.zeros((N, nx))  # true state (after adding process noise)
y_meas = np.zeros((N, ny))  # measured output

x_true = x_true_model + process_noise
for i in range(0, N):
    y_meas[i, :] = H @ x_true[i, :] + measurements_noise[i, :]


"""
###############################################################################
##########################      RTSKF Algorithm     ###########################
###############################################################################
"""


def RTSKF(x, Px, R, Q, y):
    """
    Implements Robust Two Stage Kalman filter (RTSKF) algorithm.

    Args:
        x: State estimate in the previous time step.
        Px: Initial state error covariance matrix.
        R: Measurement noise covariance matrix.
        Q: Process noise covariance matrix.
        y: Measurement vector.

    Returns:
        x_next: Updated state estimate.
        Px_next: Updated state error covariance matrix.
        d_hat: Estimated unknown input vector.
        Pd: Unknown input estimation covariance matrix.
    """

    xp = A @ x
    Px_p = A @ Px @ A.T + Q
    y_tilde = y - H @ xp

    C = H @ Px_p @ H.T + R
    Kx = Px_p @ H.T @ np.linalg.inv(C)
    x_bar = xp + Kx @ y_tilde
    Px_bar = Px_p - Kx @ H @ Px_p

    Pd = np.linalg.inv(E.T @ H.T @ np.linalg.inv(C) @ H @ E)
    Kd = Pd @ E.T @ H.T @ np.linalg.inv(C)

    d_hat = Kd @ y_tilde

    V = E - Kx @ H @ E

    x_next = x_bar + V @ d_hat
    Px_next = Px_bar + V @ Pd @ V.T

    return x_next, Px_next, d_hat, Pd


# initial estimation
x_hat = np.array([0, 0, 0])
# initial estimation error covariance matrix
Px = 10 * np.eye(3)
# sequence of estimations.
x_estimations = np.zeros((N, nx))  # state estimations
d_estimations = np.zeros((N, nd))  # unknown input estimation

for i in range(1, N):
    [x_hat, Px, d_hat, Pd] = RTSKF(x_hat, Px, R, Q, y_meas[i, :])
    x_estimations[i, :] = x_hat.reshape(-1)
    d_estimations[i - 1, :] = d_hat.reshape(-1)


"""
##############################################################################
###################             Results and plots          ###################      
##############################################################################
"""

start_point = 10
end_point = 90
state_estimation_RMSE = np.sqrt(
    np.mean(
        np.square(
            x_estimations[start_point:end_point, :]
            - x_true[start_point:end_point, :]
        )
    )
)
unknown_input_estimation_RMSE = np.sqrt(
    np.mean(
        np.square(
            d_estimations[start_point:end_point, :]
            - d_true[start_point:end_point, :]
        )
    )
)

print("state_estimation_RMSE:", state_estimation_RMSE)
print("unknown_input_estimation_RMSE:", unknown_input_estimation_RMSE)


for i in range(nx):
    plt.figure(i)
    plt.plot(x_estimations[:, i])
    plt.plot(x_true[:, i])
    plt.legend(["RTSKF", "true"])
    plt.title("state estimation " + str(i))
    plt.show

for i in range(nd):
    plt.figure(i + 3)
    plt.plot(d_estimations[:, i])
    plt.plot(d_true[0 : N - 1, i])
    plt.legend(["RTSKF", "true"])
    plt.title("unknown input estimation " + str(i))
    plt.show
