from math import sqrt
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def tr(dt, Nt,D,N0,Nx,dx,p=0.5):
	record = np.array([])
	for iNt in range(1, Nt):
		r = np.random.binomial(1, p, (iNt, N0))
		r =  2 * (r - 0.5)
		out = np.empty(r.shape)
		np.cumsum(r, axis=0, out=out)
		line = np.where(out >=Nx)[1]
		out_r1 = np.delete(out, line, axis=1)

		out_r2 = out_r1[-1,:]
		out_r3 = np.delete(out_r2, np.where(out_r2 < 0))
		record = np.append(record, out_r3)

	return record*dx

def gaussian(T,Nt,D,N0,r,dx):

	gaussian = np.zeros((2*r + 1, Nt))
	xgrid = dx * np.arange(-r, r+1, 1)

	for i in range(Nt):
		t = (i+1)*T/Nt
		sigma = np.sqrt(2*t*D)
		gaussian[:,i] = N0 * norm.pdf(xgrid, loc=0, scale=sigma)

	return xgrid, gaussian	
		
	
def main():

	# Number of steps.
	dt = 1
	D  = 0.44
	alpha = 20
	L = 10
	dx = 1
	Nx = int(L/dx)
	N0 = dt * alpha

#	plt.title('chemical random walk trajectory')
#	for i in range(N0):
#		t = range(Nt)
#		plt.plot(t, out[:,i], '-', lw=0.5, alpha=0.8)	
#
#	plt.xlabel('t/msec')
#	plt.ylabel('x/um')
#	plt.grid()
#	plt.savefig('random_walk.png')

	x_steady = np.arange(0,Nx+1,1.0)
	x_steady *= dx	

	plt.title('diffusion histogram and steady state')
	for Nt in [10, 100,4000]:
		out = tr(dt, Nt,D,N0,Nx,dx,p=0.5)
		plt.hist(out, alpha = 0.5, label='t=%s'%str(Nt))#, density=True)

	y_steady = -alpha*x_steady/D + alpha*L/D
	plt.plot(x_steady, y_steady, '-', label='steady state')	

	plt.xlabel('x/um')
	plt.ylabel('Number of particles')
	plt.legend()
	plt.grid()
	plt.show()
	plt.savefig('diffusion1.png')



if __name__ == "__main__":
	main()


