
import numpy as np
from scipy.optimize import check_grad

from dactim_angio.metrics import (
	opposite_correlation,
	opposite_mutual_information_kernel,
)


def generic(metric,metric_args_list=[]):

	x0=np.random.rand(100)
	y0=np.random.rand(100)

	func_to_test=opposite_correlation

	def func_x(x):
		return func_to_test(x,y0)[0]
	def grad_func_x(x):
		return func_to_test(x,y0)[1]

	def func_y(y):
		return func_to_test(x0,y)[0]
	def grad_func_y(y):
		return func_to_test(x0,y)[2]


	epsilons=[10**(-i) for i in range(3,7)]
	err_grad_x=[]
	for eps in epsilons:
		print()
		err_grad_x.append(check_grad(func_x,grad_func_x,x0,epsilon=eps))

	err_grad_y=[]
	for eps in epsilons:
		err_grad_y.append(check_grad(func_y,grad_func_y,y0,epsilon=eps))


	order_x=(np.log(err_grad_x[1:])-np.log(err_grad_x[:-1]))/(np.log(epsilons[1:])-np.log(epsilons[:-1]))
	order_y=(np.log(err_grad_y[1:])-np.log(err_grad_y[:-1]))/(np.log(epsilons[1:])-np.log(epsilons[:-1]))

	assert(np.mean(order_x)>=0.9)
	assert(np.mean(order_y)>=0.9)
	


def test_opposite_correlation():
	fun_list=[opposite_correlation,opposite_mutual_information_kernel]
	for f in fun_list:
		generic(opposite_correlation)
	
