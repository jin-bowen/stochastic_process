from math import sqrt
from scipy.stats import norm
import numpy as np
import numpy
from pylab import plot, show, grid, xlabel, ylabel, title, suptitle

def drifted_brownian(x0, n, dt, mu, out=None):

    x0 = np.asarray(x0)
    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), loc=mu*dt, scale=sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)
    return out

if __name__ == "__main__":

    # Total time.
    T = 10.0
    # Number of steps.
    N = 500
    # Time step size
    dt = T/N
    # Number of realizations to generate.
    m = 5
    # Create an empty array to store the realizations.
    x = numpy.empty((m,N+1))
    x_drifted = numpy.empty((m,N+1))
    # Initial values of x.
    x[:, 0] = 0
    x_drifted[:, 0] = 0

    drifted_brownian(x[:,0], N, dt, mu=0, out=x[:,1:])
    drifted_brownian(x[:,0], N, dt, mu=1, out=x_drifted[:,1:])

    t = numpy.linspace(0.0, N*dt, N+1)
    for k in range(m):
        plot(t, x[k])
    xlabel('t', fontsize=12)
    ylabel('x', fontsize=12)
    suptitle("standard BM", size=16)
    title("t=10s, dt=1/50", size=12)
    grid(True)
    show()

    for k in range(m):
        plot(t, x_drifted[k])
    xlabel('t', fontsize=12)
    ylabel('x', fontsize=12)
    suptitle("drifted BM with u=1", size=16)
    title("t=10s, dt=1/50", size=12)
    grid(True)
    show()
