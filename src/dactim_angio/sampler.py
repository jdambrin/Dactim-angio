from dactim_angio.RBF import GaussianRBF
from dactim_angio.geom import PointCloud
from dactim_angio.geom import RayFan
from dactim_angio.geom import X_to_listpoints
from dactim_angio.geom import getAffineDerivatives
from dactim_angio.geom import getAffine
from dactim_angio.geom import applyAffine
from skimage.morphology import disk, binary_dilation



import numpy as np

import matplotlib.pyplot as plt

class Sampler_Simpler():
	def __init__(self,grid_xa,grid_tof,field):


		self.grid_xa=grid_xa
		self.grid_tof=grid_tof

		self.center=grid_xa.center

		self.theta_x=0
		self.theta_y=0
		self.theta_z=0
		self.sx=0
		self.sy=0
		self.sz=0


		scalars_xa=grid_xa.dat[field[0]].flatten()

		self.scalars_xa=scalars_xa

		points_xa=X_to_listpoints(grid_xa.getCoordinates())


		h_tof=grid_tof.getH().max()

		grid_tof_rescaled=grid_tof.copy()

		scalars_tof=grid_tof_rescaled.dat[field[1]].flatten()
		points_tof=X_to_listpoints(grid_tof_rescaled.getCoordinates())

		sample_idx_tof=np.argwhere(np.abs(grid_tof_rescaled.dat[field[1]]).flatten()>0.01*np.abs(grid_tof_rescaled.dat[field[1]]).max()).flatten()
		
		scalars_tof_reduced=scalars_tof[sample_idx_tof]
		points_tof_reduced=points_tof[sample_idx_tof,:]


		self.TOF_RBF=GaussianRBF(points_tof_reduced,h_tof,{0:scalars_tof_reduced})
		self.sample_points=points_xa
		self.sample_rays=RayFan(self.center,self.sample_points)


		self.tof_init_points=points_tof_reduced.copy()

		
		foo,foo,sample_restricted_tof_projected=self.sample_nocontrast()

		self.tof_max=sample_restricted_tof_projected.max()

	def contrast(self,u):
		a=0.1*self.tof_max
		return (np.atan(u/a)*2/np.pi)*(u>0)

	def contrast_derivative(self,u):
		a=0.1*self.tof_max
		return (1/(1+(u/a)**2))*(2/(np.pi*a))*(u>0)

	def sample_nocontrast(self):
		sample_points=self.sample_points
		sample_scalar_xa=self.scalars_xa
		sample_scalar_tof=self.TOF_RBF.eval_rays(self.sample_rays)
		return sample_points,sample_scalar_xa,sample_scalar_tof

	def sample(self):
		sample_points=self.sample_points
		sample_scalar_xa=self.scalars_xa  #self.scalar_xa_rays.eval(sample_points,0)
		sample_scalar_tof=self.contrast(self.TOF_RBF.eval_rays(self.sample_rays))
		return sample_points,sample_scalar_xa,sample_scalar_tof


	def euler_gradient_sample(self):

		derivT=getAffineDerivatives(self.theta_x,self.theta_y,self.theta_z,self.sx,self.sy,self.sz)
		T,Tinv=getAffine(self.theta_x,self.theta_y,self.theta_z,self.sx,self.sy,self.sz)
				
		deriv_samples_wrt_tof=[]
		deriv_samples_wrt_xa=[]

		d_point_eval_rays=self.TOF_RBF.d_points_eval_rays(self.sample_rays)
		eval_xa_samples=self.TOF_RBF.eval_rays(self.sample_rays)
		
		for i in range(6):

			DTpoints=applyAffine(derivT[i],applyAffine(Tinv,self.TOF_RBF.point_cloud.points))

			deriv_xa_samples_wrt_tof=d_point_eval_rays[0]@DTpoints[:,0]+d_point_eval_rays[1]@DTpoints[:,1]+d_point_eval_rays[2]@DTpoints[:,2]
			deriv_xa_samples_wrt_tof=deriv_xa_samples_wrt_tof*self.contrast_derivative(eval_xa_samples)
			deriv_samples_wrt_tof.append(deriv_xa_samples_wrt_tof)

			deriv_xa_samples_wrt_xa=0*deriv_xa_samples_wrt_tof
			deriv_samples_wrt_xa.append(deriv_xa_samples_wrt_xa)

		return deriv_samples_wrt_xa,deriv_samples_wrt_tof

	def update_tof_points(self,theta_x,theta_y,theta_z,sx,sy,sz):
		self.theta_x=theta_x
		self.theta_y=theta_y
		self.theta_z=theta_z
		self.sx=sx
		self.sy=sy
		self.sz=sz
		T=getAffine(theta_x,theta_y,theta_z,sx,sy,sz)[0]
		# le tree de self.scalar_tof_cloud.point_cloud n'a pas besoin d'être mis à jour car on fait des transformations rigides --> à changer
		self.TOF_RBF.point_cloud.points=applyAffine(T,self.tof_init_points)




