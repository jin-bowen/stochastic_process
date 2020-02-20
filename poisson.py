import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import random
import math

def poisson(r, jump_size, total_t, sample_point=1000):

	points = np.zeros((sample_point, 2))
	points[:,0] = np.linspace(0, total_t, num=sample_point)
	points[:,1] = 0

	dt = float(total_t - 0)/float(sample_point - 1)
	cum_t = 0
	while(cum_t <= total_t): 	

		lambda_i = r * (3 + np.cos(cum_t) + 2 * np.cos(2*cum_t))

		p = random.random()
		interval = -math.log(1.0 - p)/lambda_i
		cum_t += interval

		nt = int(np.floor(min(cum_t,total_t)/dt))
		points[nt:,1] += jump_size
	return points


def main():

	poisson01 = poisson(0.1, 1, total_t=20, sample_point=100)
	poisson1  = poisson(1.0, 1, total_t=20, sample_point=100)
	poisson10 = poisson(10.0,1, total_t=20, sample_point=100)

	fig, ax = plt.subplots(nrows=3, sharex=True)
	plt.suptitle('counting process')

	ax[0].set_title('sample points: 1000, r = 0.1')
	ax[0].set_ylabel('N(t)')	
	ax[0].plot(poisson01[:,0], poisson01[:,1], 'g-')

	ax[1].set_title('sample points: 1000, r = 1')
	ax[1].set_ylabel('N(t)')	
	ax[1].plot(poisson1[:,0],  poisson1[:,1], 'y-')

	ax[2].set_title('sample points: 1000, r = 10')
	ax[2].set_ylabel('N(t)')	
	ax[2].plot(poisson10[:,0], poisson10[:,1], 'r-')

	plt.legend()
	plt.xlabel('time')
	plt.show()
	#ax.savefig('poisson.png')	

if __name__ == "__main__":
	main()





