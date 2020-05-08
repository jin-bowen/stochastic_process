import csv
import numpy as np
import scipy as sp
import scipy.stats as st
import matplotlib.pyplot as plt

def main():

	params = [[0.6,0.3,10,1],[0.3,0.6,10,1],[6, 3, 10,1],[3, 6, 10, 1]]

	fig, axs = plt.subplots()
	fig.suptitle('theoretical distribution for mRNA concentration')
	for param in params:

		kp = param[0]
		kn = param[1]
		r =  param[2]
		gamma = param[3]
	
		x = np.linspace(0,r/gamma,100)
	
		betaab = sp.special.beta(kp/gamma, kn/gamma)
		C = gamma / ( np.power(r,(kp+kn)/gamma ) * betaab )
		p0 = C * np.power(gamma * x, kp/gamma - 1) * np.power(r - gamma * x, kn/gamma)
		p1 = C * np.power(gamma * x, kp/gamma ) * np.power(r - gamma * x, kn/gamma - 1)
		p = p0 + p1
		axs.plot(x/np.max(x), p, '-', label='k+=%s;k-=%s;r=%s,gamma=%s'%(kp, kn, r, gamma))

	axs.legend()
	axs.grid()
	axs.set_ylabel('probability density')
	axs.set_xlabel('mRNA concentration')
	plt.savefig('pb.png')

if __name__ == "__main__":
	main()

