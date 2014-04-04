 #!/usr/bin/env python

"""Import the packages that are used"""
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial

"""Define all the parameters nesssesary to peform the MC simulation"""
steps = 400     # the number of MC steps
N = 150          # number of atoms that is placed
N_theta = 6      # number of thetas that is chosen
N_max = 2*steps
epsilon = 0.25
sigma = 0.8; sigma6 = sigma**6; sigma12 = sigma**12 # constant for the LJ-potential

cnt = np.zeros(N)
x_new = np.zeros((N_theta, 2))
x = np.zeros((N, 2))
end2end = np.zeros((N, N_max))
W_array = np.zeros((N, N_max))
x[1, 0] = 1.0  # the second atom at (1, 0)


def mc(x, L, W):
    """Perform an MC step.

    Parameters:
    -----------
    x : polymer
    L : length
    W : weight
    """
    # generate an equally spaced vector with intevals of 2pi/N_theta + a random off-set
    theta = 2 * np.pi * np.random.rand() + np.linspace(0, 2*np.pi, N_theta)
    # the positions of the new atom that is placed
    x_new[:, 0] = x[L, 0] + np.cos(theta)
    x_new[:, 1] = x[L, 1] + np.sin(theta)
    # calculate the weights of the new positions
    w = np.sum(np.exp(-E(x, x_new, L)), axis=0)
    w_sum = np.sum(w)
    w_norm = w / w_sum
    W *= w_sum/(0.75*N_theta)
    # choose a theta with a roulette wheel algoritm, this gives the index
    theta_idx = np.digitize(np.random.rand(1), np.cumsum(w_norm))
    # place the next atom at the theta that was found
    x[L+1, :] = x_new[theta_idx, :]
    # calculate the end to end distance
    end2end_abs = x[0, :] - x[L+1, :]
    end2end[L+1, cnt[L+1]] = np.sum(end2end_abs**2)
    # put the weight in an array
    W_array[L+1, cnt[L+1]] = W
    W_sum = np.sum(W_array[L+1, :cnt[L+1]])
    W_0 = np.sum(W_array[2, :cnt[2]]) # weight of the first few walkers
    # set the upper and lower limit
    up = 2.0*W_sum/W_0
    low = 1.2*W_sum/W_0
    W_new = W
    cnt[L+1] += 1
    if cnt[L+1] > N_max-2 and L < N-2:
        mc(x, L+1, W_new)
    elif L < N-2:
        if W > up: # Enrich
            W_new = W*0.5
            mc(x, L+1, W_new)   # clone 1
            W_new = W*0.5
            mc(x, L+1, W_new)   # clone 2
        elif W < low: # Prune
            W_new = 2*W
            if np.random.rand(1) < 0.5: # kill it with 50% chance
                mc(x, L+1, W_new)
        else:
            mc(x, L+1, W_new)

def E(x, x_new, L):
    """ The lennnard-Jones potential.

    Parameters:
    -----------
    x : polymer
    x_new : 6 new proposed positions for theta
    L : length
    """
    r = scipy.spatial.distance.cdist(x[0:L, :], x_new, 'euclidean')
    return 4*epsilon*(sigma12/(r**12)-sigma6/(r**6))


W = 1
for i in range(steps):
    mc(x, 1, W)

