from math import sqrt
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

def tr(T,Nt,D,N0):

	sigma = np.sqrt(2*D*T/Nt)
	# For each element of x0, generate a sample of n numbers from a
	r = norm.rvs(size=(Nt,N0), loc=0, scale=sigma)

	out = np.empty(r.shape)
	np.cumsum(r, axis=0, out=out)

	return out

def gaussian(T,Nt,D,N0,r,dx):

	gaussian = np.zeros((2*r + 1, Nt))
	xgrid = dx * np.arange(-r, r+1, 1)

	for i in range(Nt):
		t = (i+1)*T/Nt
		sigma = np.sqrt(2*t*D)
		gaussian[:,i] = N0 * norm.pdf(xgrid, loc=0, scale=sigma)

	return xgrid, gaussian	
		
	
def main():

	# Total time(msec).
	T  = 1000
	# Number of steps.
	Nt = 1000
	N0 = 1000
	D  = 0.44
	dx = np.sqrt(2*T*D/Nt) 
	out = tr(T,Nt,D,N0)
	r = max(np.min(out),np.max(out),key=abs) /dx
	r = abs(int(r))

	xgrid, gauss = gaussian(T,Nt,D,N0,r,dx)
#	plt.title('chemical random walk trajectory')
#	for i in range(N0):
#		t = range(Nt)
#		plt.plot(t, out[:,i], '-', lw=0.5, alpha=0.8)	
#
#	plt.xlabel('t/msec')
#	plt.ylabel('x/um')
#	plt.grid()
#	plt.savefig('random_walk.png')

	plt.title('chemical random walk histogram and gaussian fitting')
	for i in range(0, Nt, 10):
	
		plt.hist(out[i,:], bins = xgrid, alpha = 0.5)
		plt.plot(xgrid, gauss[:,i], '-', alpha=0.8)	

	plt.xlabel('x/um')
	plt.ylabel('Number of particles')
	plt.grid()
	plt.savefig('random_walk2.png')



if __name__ == "__main__":
	main()


