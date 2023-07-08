# RTSKF-SystemsWithUnknownInputs
This repository provides a simple implementation of the Robust Two-Stage Kalman Filters (RTSKF) for Systems with Unknown Inputs.  The code was developed from scratch due to the lack of existing straightforward implementations available online. It serves as a valuable resource for those in search of a simple RTSKF that may not be readily accessible elsewhere on the internet (to the best of the author's knowledge).

Here I add a link to the original paper for the mathematics behind and the algorithm explaination.
https://ieeexplore.ieee.org/document/895577

## The considered discrete linear model

The linear discrete stochastic system with unknown inputs is in the following form:
$$x_{k+1}  =A_k x_k+E_k d_k+w_k$$
$$y_k  =H_k x_k+\eta_k$$
where $x_k \in R^n$ is the system state, $d_k \in R^p$ is the unknown inputs, and $y_k \in R^m$ is the measurement vector. Matrices $A_k, H_k$, and $E_k$ have appropriate dimensions with the assumption that rank $(E_k)=p$, rank $(H_k)=m(\geq p)$, and rank $(H_k E_{k-1})=p$. The process noise $w_k$ and the measurement noise $\eta_k$ are white noise distributions with zero-means and covariance matrices of $Q_k$ and $R_k$, respectively.

## The simulation example
The following simulation example was conducted (the same exists in the original paper). 
```math
\begin{aligned}
& A_k=\left[\begin{array}{ccc}
0.1 & 0.5 & 0.08 \\
0.6 & 0.01 & 0.04 \\
0.1 & 0.7 & 0.05
\end{array}\right], \quad Q_k=\left[\begin{array}{ccc}
10 & 0 & 0 \\
0 & 10 & 0 \\
0 & 0 & 10
\end{array}\right] \\
& H_k=\left[\begin{array}{lll}
1 & 1 & 0 \\
0 & 1 & 1
\end{array}\right], \quad E_k=\left[\begin{array}{l}
0 \\
2 \\
1
\end{array}\right], \quad R_k=\left[\begin{array}{cc}
20 & 0 \\
0 & 20
\end{array}\right] .
\end{aligned}
```

The initial state is $x_0=[1\ 1\ 1]^T$. The initial estimate of the state is assumed to be zero, and its covariance is given by $P_0^x=diag(10,10,10)$. The unknown input is given by $$d_k=50 u_s(k)-100 u_s(k-20)$$
where $u_s[k]$ is the unit-step function.
