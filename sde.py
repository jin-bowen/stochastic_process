from math import sqrt
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def approx(dt, N, mu, sigma, x0=0.0):

	r = norm.rvs(size=N, loc=0, scale=1)
	w_e = np.zeros(N)
	w_m = np.zeros(N)
	w_e[0] = x0
	w_m[0] = x0
	delta_w =  r * sqrt(dt)

	i = 1
	while(i < N):
		w_m[i] = w_m[i-1] * (1.0 + mu * dt + sigma * delta_w[i-1]) + \
			0.5 * sigma * (delta_w[i-1] * delta_w[i-1] - dt)
		w_e[i] = w_e[i-1] * (1.0 + mu * dt +  sigma * delta_w[i-1])
	
		i = i + 1 

	r_e = r * dt
	w_exact = np.empty(r_e.shape)
	np.cumsum(r_e, axis=-1, out=w_exact)
	w_exact_out = list(map(lambda x: np.exp(x),w_exact))
	
	return w_exact_out, w_exact, w_e, w_m


if __name__ == "__main__":

	# Total time.
	T = 10.0
	# Number of steps.
	N = 100
	# Time step size
	dt = T/N

	t = np.linspace(0.0, N*dt, N)
	ax  = plt.figure()
	plt.suptitle('comparision between numerical solution and exact solution of SDE')
	plt.title('sample points: %s, sample time: %ss'%(N, T))
	#for i in range(10):

	X, log_X, w_e, w_m = approx(dt, N, 0.5, 1.0, x0=1.0)
	plt.plot(t, X, 'g-', label='exp(Bt)')
	#plt.plot(t, log_X, 'g-', label='Bt')
	plt.plot(t, w_e, 'y.-', label='Euler-Maruyama')
	plt.plot(t, w_m, 'r.-', label='Millstein')
	plt.legend()
	plt.xlabel('time')
	plt.ylabel('X(t)')

	plt.show()
	ax.savefig('SDE.png')





