import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import random
import math

def poisson(lambda_i, jump_size, total_t, sample_point=1000):

	points = np.zeros((sample_point, 2))
	points[:,0] = np.linspace(0, total_t, num=sample_point)
	points[:,1] = 0

	dt = float(total_t - 0)/float(sample_point - 1)
	cum_t = 0
	while(cum_t < total_t): 	
		p = random.random()
		interval = -math.log(1.0 - p)/lambda_i
		cum_t += interval

		nt = int(np.floor(min(cum_t,total_t)/dt))
		points[nt:-1,1] += jump_size
	return points

def levy(k, alpha, total_t, sample_point=1000):

	#plt.figure()
	sum_pp = np.zeros((sample_point, 2))
	sum_pp[:,0] = np.linspace(0, total_t, num=sample_point)

	jsize_max = 10.0
	jsize_min = -10.0
	dx = (jsize_max - jsize_min)/k
	for jsize in np.linspace(jsize_min, jsize_max, k):
		if jsize == 0: continue
		lambda_k =  dx/np.power(abs(jsize), 1.0 + alpha ) 
		pp_k = poisson(lambda_k, jsize, total_t, sample_point)
		#plt.plot(pp_k[:,0], pp_k[:,1])
		sum_pp[:,1] = sum_pp[:,1] + pp_k[:,1]
	#plt.show()
	return sum_pp

def CM_levy(alpha, total_t, sample_point):

	vi = np.random.exponential(scale=1.0, size=sample_point)
	ui = np.random.uniform(-np.pi/2.0, np.pi/2.0, size=sample_point)
	si = []
	s = 0
	for v,u in zip(vi, ui):
		s1 = np.sin(alpha * u) / np.power(abs(np.cos(u)), 1.0/alpha)
		s2 = np.power( np.cos(u-alpha*u)/v, (1.0-alpha)/alpha ) 
		s  += s1*s2	
		si.append(s*np.power(total_t/sample_point, 1.0/alpha))	
	return np.linspace(0, total_t, num=sample_point), np.array(si)	

def main():

	total_pp_05  = levy(k=1000, alpha=0.5, total_t=2.0, sample_point=1000)
	total_pp_10  = levy(k=1000, alpha=1.1, total_t=2.0, sample_point=1000)
	total_pp_15  = levy(k=1000, alpha=1.5, total_t=2.0, sample_point=1000)

	ts_05,total_pp2_05 = CM_levy(0.5, total_t=2.0, sample_point=100)
	ts_10,total_pp2_11 = CM_levy(1.1, total_t=2.0, sample_point=100)
	ts_15,total_pp2_15 = CM_levy(1.5, total_t=2.0, sample_point=100)

	ax = plt.figure()
	plt.suptitle('mixed poisson approximation')
	plt.title('sample points: 1000, number of poisson process: 1000')
	plt.plot(total_pp_05[:,0], total_pp_05[:,1], 'g-', label='lambda=0.5')
	plt.plot(total_pp_10[:,0], total_pp_10[:,1], 'y-', label='lambda=1.1')
	plt.plot(total_pp_15[:,0], total_pp_15[:,1], 'r-', label='lambda=1.5')
	plt.legend()
	plt.xlabel('time')
	plt.ylabel('s(t)')
	ax.savefig('poisson.png')
	
	ax = plt.figure()
	plt.suptitle('chamber-mallow algorithm')
	plt.title('sample points: 1000')
	plt.plot(ts_05, total_pp2_05, 'g-', label='lambda=0.5')
	plt.plot(ts_10, total_pp2_11, 'y-', label='lambda=1.1')
	plt.plot(ts_15, total_pp2_15, 'r-', label='lambda=1.5')
	plt.legend()
	plt.xlabel('time')
	plt.ylabel('s(t)')
	ax.savefig('CM.png')

if __name__ == "__main__":
	main()





