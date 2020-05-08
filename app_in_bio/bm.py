from math import sqrt
from scipy.stats import norm
import numpy as np
import numpy
from pylab import plot, show, grid, xlabel, ylabel, title, suptitle

def brownian(x0, n, dt, out=None):

	x0 = np.asarray(x0)
	# For each element of x0, generate a sample of n numbers from a
	# normal distribution.
	r = norm.rvs(size=x0.shape + (n,), loc=0, scale=sqrt(dt))

	# If `out` was not given, create an output array.
	if out is None:
		out = np.empty(r.shape)

	# This computes the Brownian motion by forming the cumulative sum of
	# the random samples.
	np.cumsum(r, axis=-1, out=out)

	# Add the initial condition.
	out += np.expand_dims(x0, axis=-1)
	return out

def integral(bm, dt):
	N = bm.shape[0]
	#anti_ito
	integral1 = 0
	integral2 = 0
	integral3 = 0
	
	for i in range(1, N):
		integral1 += bm[i] * (bm[i] - bm[i-1])
		integral2 += 0.5 * (bm[i] - bm[i-1]) * (bm[i] - bm[i-1])
		integral3 += bm[i-1] * (bm[i] - bm[i-1])

	return integral1, integral2, integral3


if __name__ == "__main__":

	# Total time.
	T = 1
	# Number of steps.
	N = 100
	# Time step size
	dt = T/N
	# Create an empty array to store the realizations.
	out = numpy.empty(N+1)
	# Initial values of x.
	out[0] = 0

	brownian(out[0], N, dt, out=out[1:])
	integral1, integral2, integral3 = integral(out, dt)

	exact1 = 0.5 * out[-1]*out[-1] + 0.5 * T
	exact2 = 0.5 * out[-1]*out[-1] 
	exact3 = 0.5 * out[-1]*out[-1] - 0.5 * T

	print(integral1, exact1)
	print(integral2, exact2)
	print(integral3, exact3)

	t = numpy.linspace(0.0, N*dt, N+1)
	plot(t, out)
	xlabel('t', fontsize=12)
	ylabel('x', fontsize=12)
	suptitle("standard BM", size=16)
	title("t=1s, dt=0.01", size=12)
	grid(True)
	show()


