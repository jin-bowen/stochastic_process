import numpy as np
import matplotlib.pyplot as plt

def f(points, params):

	ka,kb,ra,rb,ua,ub,omega = params

	xa, xb = points
	k1 = (xa + 2.0*xb)/(ka/omega + xa + 2.0*xb)
	k2 = (xb + 2.0*xa)/(kb/omega + xb + 2.0*xa)

	da = -1.0 * ua * xa + ra *(1.0 - k1) * xa
	db = -1.0 * ub * xb + rb *(1.0 - k2) * xb

	return(np.array([da, db]))

def seed(params):
	ka,kb,ra,rb,ua,ub,omega = params

	y_seed = 1/(3*omega) * ( 2*ka*ra/ua - kb*rb/ub - 2*ka + kb)
	x_seed = 1/(3*omega) * ( 2*kb*rb/ub - ka*ra/ua - 2*kb + ka)

	return x_seed, y_seed

def main():
	A = np.linspace(0, 4, 20)
	B = np.linspace(0, 4, 20)

	ka = 50
	kb = 40
	ra = 3
	rb = 4
	ua = ub =1
	omega = 20
	params = [ka,kb,ra,rb,ua,ub,omega]

	A1, B1 = np.meshgrid(A, B)
	DA1, DB1 = f([A1, B1], params)
	x_seed, y_seed = seed(params)

	fig, axs = plt.subplots()
	axs.quiver(A1, B1, DA1, DB1, color='r')
	axs.plot( x_seed, y_seed, '.', color='green', marker='o', label='steady state' )
	
	axs.set_xlabel('xA')
	axs.set_ylabel('xB')
	axs.set_title('system1')
	plt.savefig('system1_ODE.png')


if __name__ == "__main__":
	main()





