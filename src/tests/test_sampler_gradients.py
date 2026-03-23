import sys

from dactim_angio.grid import SpatialGrid
from dactim_angio.grid import SpatialGridAffine
from dactim_angio.grid import SpatialGrid_XA

from dactim_angio.geom import Camera
from dactim_angio.geom import GaussianRays
from dactim_angio.geom import GaussianPoints
from dactim_angio.geom import RayFan
from dactim_angio.geom import X_to_listpoints
from dactim_angio.geom import cart2sphere
from dactim_angio.geom import getAffineDerivatives
from dactim_angio.geom import getAffine
from dactim_angio.geom import applyAffine
from dactim_angio.geom import kernel_r

from dactim_angio.metrics import opposite_correlation
from dactim_angio.metrics import blabla
from dactim_angio.metrics import opposite_mutual_information_kernel

from scipy.spatial import KDTree
from scipy.sparse import coo_array
from scipy.sparse.linalg import cg
from scipy.ndimage import gaussian_filter

from scipy.optimize import check_grad

from sklearn.neighbors import BallTree
from scipy.optimize import minimize

import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt

import time


def load_data(path_xa,path_xa_metadata,path_tof,path_tof_segm,right):

	grid_xa=SpatialGrid_XA.fromNifti(path_xa,path_xa_metadata)

	#for d in grid_xa.dat:
	#	vmax=np.percentile(grid_xa.dat[d],99)
	#	vmin=0#0.2*vmax
	#	grid_xa.dat[d]=np.clip(grid_xa.dat[d],vmin,vmax)-vmin


	grid_tof=SpatialGridAffine.fromNifti(path_tof)

	grid_tof_segm=SpatialGridAffine.fromNifti(path_tof_segm)

	if(right):
		grid_tof.dat['1']=((grid_tof_segm.dat['0']==2)+(grid_tof_segm.dat['0']==4)>0).astype('float')
	else:
		grid_tof.dat['1']=((grid_tof_segm.dat['0']==1)+(grid_tof_segm.dat['0']==4)>0).astype('float')

	grid_tof.dat['2']=(grid_tof_segm.dat['0']>0).astype('float')

	return grid_xa, grid_tof


def validGradients_sampler(grid_xa, grid_tof,xa_frame):
	
	mySampler=Sampler(grid_xa,grid_tof,2,500,xa_frame)
	x0=[0,0,0,0,0,0] 

	@measure_time
	def func_to_minimize(x,*args):
		sampler=args[0]
		sampler.update_tof_points(x[0],x[1],x[2],x[3],x[4],x[5])
		sample_points,sample_scalar_xa,sample_scalar_tof=sampler.sample()
		return np.array(sample_scalar_xa)

	@measure_time
	def Jac_func_to_minimize(x,*args):
		sampler=args[0]
		sampler.update_tof_points(x[0],x[1],x[2],x[3],x[4],x[5])
		sample_points,sample_scalar_xa,sample_scalar_tof=sampler.sample()
		deriv_samples_wrt_xa,deriv_samples_wrt_tof=sampler.euler_gradient_sample()
		return np.array(deriv_samples_wrt_xa).T


	epsilons=[10**(-i) for i in range(3,9)]
	err_grad_xa=[]
	for eps in epsilons:
		err_grad_xa.append(check_grad(func_to_minimize,Jac_func_to_minimize,x0,(mySampler),epsilon=eps))


	@measure_time
	def func_to_minimize(x,*args):
		sampler=args[0]
		sampler.update_tof_points(x[0],x[1],x[2],x[3],x[4],x[5])
		sample_points,sample_scalar_xa,sample_scalar_tof=sampler.sample()
		return np.array(sample_scalar_tof)

	@measure_time
	def Jac_func_to_minimize(x,*args):
		sampler=args[0]
		sampler.update_tof_points(x[0],x[1],x[2],x[3],x[4],x[5])
		sample_points,sample_scalar_xa,sample_scalar_tof=sampler.sample()
		deriv_samples_wrt_xa,deriv_samples_wrt_tof=sampler.euler_gradient_sample()
		return np.array(deriv_samples_wrt_tof).T


	epsilons=[10**(-i) for i in range(3,9)]
	err_grad_tof=[]
	for eps in epsilons:
		err_grad_tof.append(check_grad(func_to_minimize,Jac_func_to_minimize,x0,(mySampler),epsilon=eps))

	plt.loglog(epsilons,err_grad_xa,'.-')
	plt.loglog(epsilons,err_grad_tof,'.-')
	plt.show()


def pre_proc_dsa(grid_xa):
	X=np.zeros((len(grid_xa.dat),np.prod(grid_xa.shape)))
	grid_xa_new=grid_xa.copy()

	for i in range(X.shape[0]):
		grid_xa_new.dat[i]=np.clip(grid_xa.dat[0]-grid_xa.dat[i],0,None)

	return grid_xa_new
	
if __name__=='__main__':

	grid_xa, grid_tof=load_data()

	validGradients_sampler(grid_xa, grid_tof,xa_frame)

	print(times)


