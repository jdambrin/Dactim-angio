import sys
sys.path.append('../')
from dactim_angio.grid import SpatialGrid
from dactim_angio.grid import SpatialGridAffine
from dactim_angio.grid import SpatialGridPerspective

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

from scipy.spatial import KDTree
from scipy.sparse import coo_array
from scipy.sparse.linalg import cg
from scipy.ndimage import gaussian_filter

from sklearn.neighbors import BallTree
from scipy.optimize import minimize

import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt

import time

def measure_time(func):
    def new_func(*args, **kwargs):
        tic = time.time()
        res = func(*args, **kwargs)
        toc = time.time()
        print(f"[{func.__name__} finished in {toc-tic:.4f}s]")
        return res
    return new_func



class Sampler():
	def __init__(self,grid_xa,grid_tof,sigma_add,N_samples,xa_frame):
		self.grid_xa=grid_xa
		self.grid_tof=grid_tof
		self.xa_frame=xa_frame
		self.center=grid_xa.center

		self.theta_x=0
		self.theta_y=0
		self.theta_z=0
		self.sx=0
		self.sy=0
		self.sz=0

		self.sigma_add=sigma_add

		#dist_source_detector=np.linalg.norm(X_to_listpoints(grid_xa.getCoordinates())-grid_xa.center,axis=1).min()
		dist_source_detector=np.linalg.norm(grid_xa.center)#+np.linalg.norm(points_xa_reduced,axis=1).min()

		h_xa=grid_xa.getH().max()
		step_angle_xa=0.5*np.asin(h_xa/dist_source_detector)

		if(sigma_add>0):
			rescale_factor_xa=sigma_add/h_xa
			grid_xa_rescaled=grid_xa.copy()
			grid_xa_rescaled.dat[xa_frame]=gaussian_filter(grid_xa.dat[xa_frame],rescale_factor_xa)
			grid_xa_rescaled.rescale(tuple([int(n/rescale_factor_xa) for n in grid_xa.shape]))
			h_xa=h_xa*rescale_factor_xa
			step_angle_xa=0.5*np.asin(h_xa/dist_source_detector)
		else:
			grid_xa_rescaled=grid_xa.copy()


		scalars_xa=grid_xa_rescaled.dat[xa_frame].flatten()
		points_xa=X_to_listpoints(grid_xa_rescaled.getCoordinates())
		sample_idx_xa=np.argwhere(grid_xa_rescaled.dat[xa_frame].flatten()>0.05*grid_xa_rescaled.dat[xa_frame].max()).flatten()
		scalars_xa_reduced=scalars_xa[sample_idx_xa]
		points_xa_reduced=points_xa[sample_idx_xa,:]
		rays_xa=RayFan(grid_xa.center,points_xa_reduced)
		self.scalar_xa_rays=GaussianRays(rays_xa,scalars_xa_reduced,step_angle_xa)
	

		h_tof=grid_tof.getH().max()

		if(sigma_add>0):
			rescale_factor_tof=0.5*sigma_add/h_tof
			grid_tof_rescaled=grid_tof.copy()
			grid_tof_rescaled.dat['1']=gaussian_filter(grid_tof.dat['1'],rescale_factor_tof)
			grid_tof_rescaled.rescale(tuple([int(n/rescale_factor_tof) for n in grid_tof.shape]))
			h_tof=h_tof*rescale_factor_tof
		else:
			grid_tof_rescaled=grid_tof.copy()


		scalars_tof=grid_tof_rescaled.dat['1'].flatten()
		points_tof=X_to_listpoints(grid_tof_rescaled.getCoordinates())

		sample_idx_tof=np.argwhere(grid_tof_rescaled.dat['1'].flatten()>0.01*grid_tof_rescaled.dat['1'].max()).flatten()
		scalars_tof_reduced=scalars_tof[sample_idx_tof]
		points_tof_reduced=points_tof[sample_idx_tof,:]

		self.scalar_tof_cloud=GaussianPoints(points_tof_reduced,scalars_tof_reduced,h_tof)
	
		self.N_samples=N_samples

		#self.sample_points_xa=np.random.normal(0,self.sigma_add,(self.N_samples,3))+points_xa_reduced[np.random.choice(points_xa_reduced.shape[0],self.N_samples),:]
		#self.sample_points_tof=np.random.normal(0,self.sigma_add,(self.N_samples,3))+points_tof_reduced[np.random.choice(points_tof_reduced.shape[0],self.N_samples),:]
		
		pdf_xa=scalars_xa_reduced/np.sum(scalars_xa_reduced)
		pdf_tof=scalars_tof_reduced/np.sum(scalars_tof_reduced)

		#self.N_samples=len(scalars_xa_reduced)

		self.sample_points_xa=np.random.normal(0,h_xa,(self.N_samples,3))+points_xa_reduced[np.random.choice(points_xa_reduced.shape[0],self.N_samples),:]
		self.sample_points_tof=np.random.normal(0,h_tof,(self.N_samples,3))+points_tof_reduced[np.random.choice(points_tof_reduced.shape[0],self.N_samples),:]
		
		self.sample_points_tof_init=self.sample_points_tof.copy()
		self.scalar_tof_cloud_points_init=self.scalar_tof_cloud.points.copy()

		#self.update_tof_points(self.theta_x,self.theta_y,self.theta_z,self.sx,self.sy,self.sz)

		#plt.imshow(np.max(grid_tof_rescaled.dat['1'],axis=0))
		#plt.show()

#		plot_xa_tof(grid_xa_rescaled, grid_tof_rescaled)

		#print('Ratio XA :',len(scalars_xa_reduced)/self.N_samples)
		#plot_xa_tof_with_sample_points(grid_xa_rescaled,grid_tof_rescaled,self.sample_points_xa,self.sample_points_tof)


	def sample(self):
		sample_points=np.concatenate((self.sample_points_xa,self.sample_points_tof))
		sample_rays=RayFan(self.center,sample_points)
		#sample_scalar_xa=self.scalar_xa_rays.eval(sample_points,self.sigma_add_angle)
		#sample_scalar_tof=self.scalar_tof_cloud.eval_rays(sample_rays,self.sigma_add)
		sample_scalar_xa=self.scalar_xa_rays.eval(sample_points,0)
		sample_scalar_tof=self.scalar_tof_cloud.eval_rays(sample_rays,0)
		return sample_points,sample_scalar_xa,sample_scalar_tof

	def euler_gradient_sample(self):
		derivT=getAffineDerivatives(self.theta_x,self.theta_y,self.theta_z,self.sx,self.sy,self.sz)
		sample_points=np.concatenate((self.sample_points_xa,self.sample_points_tof))
		sample_rays=RayFan(self.center,sample_points)
		sample_scalar_tof_derivative=[]
		for i in range(6):
			sample_scalar_tof_derivative.append(self.scalar_tof_cloud.eval_rays_shape_derivative(lambda x : applyAffine(derivT[i],x) ,sample_rays,0))
			#sample_scalar_tof_derivative.append(self.scalar_tof_cloud.eval_rays_shape_derivative(lambda x : applyAffine(derivT[i],x) ,sample_rays,self.sigma_add))
		return sample_scalar_tof_derivative

	def update_tof_points(self,theta_x,theta_y,theta_z,sx,sy,sz):
		self.theta_x=theta_x
		self.theta_y=theta_y
		self.theta_z=theta_z
		self.sx=sx
		self.sy=sy
		self.sz=sz
		T=getAffine(theta_x,theta_y,theta_z,sx,sy,sz)[0]
		self.sample_points_tof=applyAffine(T,self.sample_points_tof_init)
		self.scalar_tof_cloud.points=applyAffine(T,self.scalar_tof_cloud_points_init)

	def update_tof_points2(self,theta_x,theta_y,theta_z,sx,sy,sz):
		self.theta_x=theta_x
		self.theta_y=theta_y
		self.theta_z=theta_z
		self.sx=sx
		self.sy=sy
		self.sz=sz
		T=getAffine(theta_x,theta_y,theta_z,sx,sy,sz)[0]
		self.scalar_tof_cloud.points=applyAffine(T,self.scalar_tof_cloud_points_init)

def load_data(path_xa,path_tof,path_tof_segm,right):

	grid_xa=SpatialGridPerspective.fromNifti(path_xa)

	for d in grid_xa.dat:
		vmax=np.percentile(grid_xa.dat[d],99)
		vmin=0#0.2*vmax
		grid_xa.dat[d]=np.clip(grid_xa.dat[d],vmin,vmax)-vmin


	grid_tof=SpatialGridAffine.fromNifti(path_tof)

	grid_tof_segm=SpatialGridAffine.fromNifti(path_tof_segm)

	if(right):
		grid_tof.dat['1']=grid_tof.dat['0']*((grid_tof_segm.dat['0']==2)+(grid_tof_segm.dat['0']==4)>0)
	else:
		grid_tof.dat['1']=grid_tof.dat['0']*((grid_tof_segm.dat['0']==1)+(grid_tof_segm.dat['0']==4)>0)

	grid_tof.dat['2']=grid_tof.dat['0']*(grid_tof_segm.dat['0']>0)

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
		sample_scalar_tof_derivative=sampler.euler_gradient_sample()
		corr,deriv_corr= metric(sample_scalar_xa,sample_scalar_tof)
		return [np.dot(deriv_corr,sample_scalar_tof_derivative[i]) for i in range(6)]

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


def validGradients(grid_xa, grid_tof):

	@measure_time
	def func_to_minimize(x,*args):
		sampler=args[0]
		sampler.update_tof_points(x[0],x[1],x[2],x[3],x[4],x[5])
		sample_points,sample_scalar_xa,sample_scalar_tof=sampler.sample()
		return opposite_correlation(sample_scalar_xa,sample_scalar_tof)[0]

	@measure_time
	def Jac_func_to_minimize(x,*args):
		sampler=args[0]
		sampler.update_tof_points(x[0],x[1],x[2],x[3],x[4],x[5])
		sample_points,sample_scalar_xa,sample_scalar_tof=sampler.sample()
		sample_scalar_tof_derivative=sampler.euler_gradient_sample()
		corr,deriv_corr= opposite_correlation(sample_scalar_xa,sample_scalar_tof)
		return [np.dot(deriv_corr,sample_scalar_tof_derivative[i]) for i in range(6)]

	mySampler=Sampler(grid_xa,grid_tof,2,500,9)
	x0=[0,0,0,0,0,0] 
	sample_points,sample_scalar_xa,sample_scalar_tof=mySampler.sample()

	shape_deriv_tofbar=mySampler.euler_gradient_sample()

	direction=np.array([10,10,10,10,10,10])*2*(np.random.rand(6)-0.5)
	t=np.linspace(-1,1,100)

	deriv=np.dot(Jac_func_to_minimize(x0,mySampler),direction)
	u0=func_to_minimize(x0,mySampler)

	u_approx=np.zeros(100)
	u=np.zeros(100)

	for i in range(100):
		print(i)
		u_approx[i]=t[i]*deriv+u0
		u[i]=func_to_minimize(t[i]*direction,mySampler)

	plt.plot(0,u0,marker='o')
	plt.plot(t,u_approx)
	plt.plot(t,u)
	plt.show()

def testIntegralAlongRays():

	points=np.array([[1,1,0],[1,-1,0]])
	rays=np.array([[1,0,0],[0,1,0]])
	center=np.array([[0,0,0]])

	T=getAffine(10,-5,15,1,2,3)[0]
	points=applyAffine(T,points)
	rays=applyAffine(T,rays)
	center=applyAffine(T,center)


	scalar_cloud=GaussianPoints(points,np.array([1,1]),1)
	sample_rays=RayFan(center,rays)

	print(scalar_cloud.eval_rays(sample_rays))
	print(2*np.exp(-1)*np.sqrt(np.pi)*scalar_cloud.weights[0])

def testIntegralAlongRaysDerivative():

	points=np.array([[1,1,0],[-1,1,0]])
	rays=np.array([[1,-1,0],[0,0,0]])
	center=np.array([[0,-1,0]])

	T=getAffine(0,0,0,0,0,0)[0]
	points=applyAffine(T,points)
	rays=applyAffine(T,rays)
	center=applyAffine(T,center)

	scalar_cloud=GaussianPoints(points,np.array([1,1]),1)
	sample_rays=RayFan(center,rays)

	derivT=getAffineDerivatives(0,0,0,0,0,0)

	sample_scalar_tof_derivative=[]

	for i in range(6):
		sample_scalar_tof_derivative.append(scalar_cloud.eval_rays_shape_derivative(lambda x : applyAffine(derivT[i],x) ,sample_rays))

	for d in sample_scalar_tof_derivative :
		print(d)


	print(2*4*np.exp(-2**2)*np.sqrt(np.pi)*scalar_cloud.weights[0])

def plot_xa_tof(grid_xa,grid_tof,xa_frame):

	cam=Camera.fromPointClouds(grid_xa.center,-grid_xa.affine[:3,0],grid_xa.affine[:3,1],X_to_listpoints(grid_xa.getCoordinates()))

	grid_tof_pv=grid_tof.toPyvista()
	grid_xa_pv=grid_xa.toPyvista()
	cam_pv=cam.toPyvista()

	p = pv.Plotter(window_size=[2000, 2000])

	p.add_mesh(cam_pv,line_width=5,color='darkred',opacity=0.5)
	p.add_mesh(grid_xa_pv,cmap='Reds',opacity='linear',scalars=str(xa_frame),show_scalar_bar=False)
	p.add_mesh(grid_tof_pv.contour([0.1*grid_tof_pv.point_data['1'].max()],grid_tof_pv.point_data['1']), color="blue",copy_mesh=True,opacity=0.2)	
#p.add_mesh(grid_tof_pv.contour(np.linspace(0,grid_tof_pv.point_data['1'].max(),100),grid_tof_pv.point_data['1']), cmap='Blues',copy_mesh=True,opacity='linear')	
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


if __name__=='__main__':
	patient_id=22
	xa_frame=17
	
	path_tof='/Users/juliendambrine/Codes/ClementT/data_processed/PIMISUTT/tof_nifti_files/Patient_'+str(patient_id)+'.nii.gz'
	path_xa='/Users/juliendambrine/Codes/ClementT/data_processed/PIMISUTT/xa_front_nifti_files/Patient_'+str(patient_id)+'.nii.gz'
	path_tof_segm='/Users/juliendambrine/Codes/ClementT/data_processed/PIMISUTT/topcow_to_full_segmentation/Patient_'+str(patient_id)+'_CoW_seg.nii.gz'


	grid_xa, grid_tof=load_data(path_xa,path_tof,path_tof_segm,right=False)
	x=multiScaleRegister(grid_xa, grid_tof,xa_frame,sigma_add=[16,8,4,2,0],Nsamples=[125,250,500,1000,2000],metric=opposite_correlation)



	