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

from dactim_angio.sampler import Sampler


times={}

def measure_time(func):
    def new_func(*args, **kwargs):
        tic = time.time()
        res = func(*args, **kwargs)
        toc = time.time()
        #print(f"[{func.__name__} finished in {toc-tic:.4f}s]")
        #print(times)
        if func.__name__ not in times:
        	times[func.__name__] = 0
        times[func.__name__]+=toc-tic
        return res
    return new_func


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

def register(grid_xa, grid_tof,sigma_add,Nsamples,x0,xa_frame,metric):
	@measure_time
	def func_to_minimize(x,*args):
		sampler=args[0]
		sampler.update_tof_points(x[0],x[1],x[2],x[3],x[4],x[5])
		sample_points,sample_scalar_xa,sample_scalar_tof=sampler.sample()
		return metric(sample_scalar_xa,sample_scalar_tof)[0]

	@measure_time
	def Jac_func_to_minimize(x,*args):
		sampler=args[0]
		sampler.update_tof_points(x[0],x[1],x[2],x[3],x[4],x[5])
		sample_points,sample_scalar_xa,sample_scalar_tof=sampler.sample()
		deriv_samples_wrt_xa,deriv_samples_wrt_tof=sampler.euler_gradient_sample()
		corr,deriv_corr_x,deriv_corr_y= metric(sample_scalar_xa,sample_scalar_tof)
		return [np.dot(deriv_corr_x,deriv_samples_wrt_xa[i])+np.dot(deriv_corr_y,deriv_samples_wrt_tof[i]) for i in range(6)]

	mySampler=Sampler(grid_xa,grid_tof,sigma_add,Nsamples,xa_frame)

	def cb(x):
		#mySampler.update_tof_points(x[0],x[1],x[2],x[3],x[4],x[5])
		#sample_points,sample_scalar_xa,sample_scalar_tof=mySampler.sample()
		#plt.scatter(sample_points[:mySampler.N_samples,1],sample_points[:mySampler.N_samples,2],color='red',alpha=0.5)
		#plt.scatter(sample_points[mySampler.N_samples:,1],sample_points[mySampler.N_samples:,2],color='blue',alpha=0.5)
		#plt.show()
		#plt.scatter(sample_scalar_xa,sample_scalar_tof)
		#plt.show()

		print(x)
	
	res=minimize(func_to_minimize,
		x0,
		args=(mySampler),
		method='L-BFGS-B',
		jac=Jac_func_to_minimize, 
		bounds=[(-20,20),(-20,20),(-20,20),(-30,30),(-30,30),(-30,30)],
		callback= cb)

	print(res)
	return res.x

def multiScaleRegister(grid_xa, grid_tof,xa_frame,sigma_add,Nsamples,metric):
	
	plot_xa_tof(grid_xa, grid_tof,xa_frame)

	#validGradients(grid_xa, grid_tof)

	grid_tof_plot=grid_tof.copy()	
	affine_init=grid_tof.affine.copy()

	x0=[0,0,0,0,0,0]

	for i in range(len(sigma_add)):
		x=register(grid_xa, grid_tof,sigma_add[i],Nsamples[i],x0,xa_frame,metric)
		x0=x
		
	T=getAffine(x[0],x[1],x[2],x[3],x[4],x[5])[0]
	grid_tof_plot.affine=T@affine_init
	plot_xa_tof(grid_xa, grid_tof_plot,xa_frame)
	
	return x


def validGradients_metric(metric,metric_args_list=[]):
	pass

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

def validGradients_full_pipeline(grid_xa, grid_tof,xa_frame,metric,metric_args_list=[]):

	@measure_time
	def func_to_minimize(x,*args):
		sampler=args[0]
		sampler.update_tof_points(x[0],x[1],x[2],x[3],x[4],x[5])
		sample_points,sample_scalar_xa,sample_scalar_tof=sampler.sample()
		return metric(sample_scalar_xa,sample_scalar_tof,metric_args_list)[0]

	@measure_time
	def Jac_func_to_minimize(x,*args):
		sampler=args[0]
		sampler.update_tof_points(x[0],x[1],x[2],x[3],x[4],x[5])
		sample_points,sample_scalar_xa,sample_scalar_tof=sampler.sample()
		deriv_samples_wrt_xa,deriv_samples_wrt_tof=sampler.euler_gradient_sample()
		corr,deriv_corr_x,deriv_corr_y= metric(sample_scalar_xa,sample_scalar_tof,metric_args_list)
		return [np.dot(deriv_corr_x,deriv_samples_wrt_xa[i])+np.dot(deriv_corr_y,deriv_samples_wrt_tof[i]) for i in range(6)]

	mySampler=Sampler(grid_xa,grid_tof,2,500,xa_frame)
	
	x0=0.5*np.array([1,1,1,1,1,1])*2*(np.random.rand(6)-0.5)

	sample_points,sample_scalar_xa,sample_scalar_tof=mySampler.sample()

	shape_deriv_tofbar=mySampler.euler_gradient_sample()

	direction=0.5*np.array([1,1,1,1,1,1])*2*(np.random.rand(6)-0.5)
	t=np.linspace(-1,1,101)

	deriv=np.dot(Jac_func_to_minimize(x0,mySampler),direction)
	u0=func_to_minimize(x0,mySampler)

	u_approx=np.zeros(101)
	u=np.zeros(101)

	for i in range(101):
		print(i)
		u_approx[i]=t[i]*deriv+u0
		u[i]=func_to_minimize(t[i]*direction+x0,mySampler)

	plt.plot(0,u0,marker='o')
	plt.plot(t,u_approx)
	plt.plot(t,u)
	plt.show()

def plot_xa_tof(grid_xa,grid_tof,xa_frame):

	cam=Camera.fromPointClouds(grid_xa.center,-grid_xa.affine[:3,0],grid_xa.affine[:3,1],X_to_listpoints(grid_xa.getCoordinates()))

	grid_tof_pv=grid_tof.toPyvista()
	grid_xa_pv=grid_xa.toPyvista()
	cam_pv=cam.toPyvista()

	p = pv.Plotter(window_size=[2000, 2000])

	p.add_mesh(cam_pv,line_width=5,color='darkred',opacity=0.5)
	p.add_mesh(grid_xa_pv,cmap='Reds',opacity='linear',scalars=str(xa_frame),show_scalar_bar=False)
	p.add_mesh(grid_tof_pv.contour([0.1*grid_tof_pv.point_data['1'].max()],grid_tof_pv.point_data['1']), color="blue",copy_mesh=True,opacity=0.2)	
	p.add_mesh(grid_tof_pv.contour([0.1*grid_tof_pv.point_data['1'].max()],grid_tof_pv.point_data['2']), color="gray",copy_mesh=True,opacity=0.1)
	
	p.camera.position = tuple(grid_xa.center)
	center=grid_xa.push(np.array([0.5*grid_xa.shape[0],0.5*grid_xa.shape[1]]))
	up=grid_xa.push(np.array([0,0.5*grid_xa.shape[1]]))-center
	dist=cam.dist
	angle=2*np.arctan(np.linalg.norm(up)/dist)*180/np.pi
	p.camera.focal_point = tuple(center)
	p.camera.up = tuple(np.array(up))
	p.camera.view_angle=angle

	p.show()

def plot_xa_tof_with_sample_points(grid_xa,grid_tof,sample_points_xa,sample_points_tof):

	cam=Camera.fromPointClouds(grid_xa.center,-grid_xa.affine[:3,0],grid_xa.affine[:3,1],X_to_listpoints(grid_xa.getCoordinates()))

	grid_tof_pv=grid_tof.toPyvista()
	grid_xa_pv=grid_xa.toPyvista()
	cam_pv=cam.toPyvista()
	points_xa_pv=pv.PolyData(sample_points_xa)
	points_tof_pv=pv.PolyData(sample_points_tof)


	p = pv.Plotter(window_size=[2000, 2000])

	p.add_mesh(cam_pv,line_width=5,color='darkred',opacity=0.5)
	p.add_mesh(grid_xa_pv,cmap='Reds',opacity='linear',scalars=str(9),show_scalar_bar=False)
	p.add_mesh(grid_tof_pv.contour([0.1*grid_tof_pv.point_data['1'].max()],grid_tof_pv.point_data['1']), color="blue",copy_mesh=True,opacity=0.2)	
	p.add_mesh(points_xa_pv,color="red", point_size=10, render_points_as_spheres=True)
	p.add_mesh(points_tof_pv,color="blue", point_size=10, render_points_as_spheres=True)


	p.camera.position = tuple(grid_xa.center)
	center=grid_xa.push(np.array([0.5*grid_xa.shape[0],0.5*grid_xa.shape[1]]))
	up=grid_xa.push(np.array([0,0.5*grid_xa.shape[1]]))-center
	dist=cam.dist
	angle=2*np.arctan(np.linalg.norm(up)/dist)*180/np.pi
	p.camera.focal_point = tuple(center)
	p.camera.up = tuple(np.array(up))
	p.camera.view_angle=angle

	p.show()


def pre_proc_dsa(grid_xa):
	X=np.zeros((len(grid_xa.dat),np.prod(grid_xa.shape)))
	grid_xa_new=grid_xa.copy()

	for i in range(X.shape[0]):
		grid_xa_new.dat[i]=np.clip(grid_xa.dat[0]-grid_xa.dat[i],0,None)

	return grid_xa_new
	
if __name__=='__main__':

	patient_id=17
	xa_frame=18
	
	path_tof='/Users/juliendambrine/Codes/ClementT/data_processed/PIMISUTT/tof_nifti_files/Patient_'+str(patient_id)+'.nii.gz'
	path_xa='/Users/juliendambrine/Codes/ClementT/data_processed/PIMISUTT/xa_front_nifti_files/Patient_'+str(patient_id)+'.nii.gz'
	path_xa_metadata='/Users/juliendambrine/Codes/ClementT/data_processed/PIMISUTT/xa_front_nifti_files/Patient_'+str(patient_id)+'_metadata.json'
	path_tof_segm='/Users/juliendambrine/Codes/ClementT/data_processed/PIMISUTT/topcow_to_full_segmentation/Patient_'+str(patient_id)+'_CoW_seg.nii.gz'

	grid_xa, grid_tof=load_data(path_xa,path_xa_metadata,path_tof,path_tof_segm,right=True)
	grid_xa=pre_proc_dsa(grid_xa)

	bounds_xa=(grid_xa.dat[xa_frame].min(),grid_xa.dat[xa_frame].max())
	bounds_tof=(0,15)

	validGradients_full_pipeline(grid_xa, grid_tof,xa_frame,opposite_mutual_information_kernel,metric_args_list=[bounds_xa,bounds_tof,50])
	#validGradients(grid_xa, grid_tof,xa_frame,opposite_correlation)
	#validGradients_sampler(grid_xa, grid_tof,xa_frame)

	#x=multiScaleRegister(grid_xa, grid_tof,xa_frame,sigma_add=[16,8,4,2,0],Nsamples=[125,250,500,1000,2000],metric=opposite_correlation)

	print(times)


