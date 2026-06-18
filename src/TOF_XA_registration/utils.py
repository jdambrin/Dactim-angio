import matplotlib.pyplot as plt
import numpy as np
import os
import json
from dactim_angio.spatial.RBF import GaussianRBF
from dactim_angio.spatial.geom import RayFan, X_to_listpoints, getAffineDerivatives, getAffine, applyAffine

class Monitor:
	def __init__(self,sampler_front,sampler_sagit,x0,components,out_png):


		self.out_png=out_png
		self.tmp_png = out_png + ".tmp"

		self.x0=x0
		self.components=components

		self.sampler_front=sampler_front
		self.sampler_sagit=sampler_sagit

		self.tabit=[0]
		self.taby=[x0]

		fig, ax = plt.subplots(2,3,figsize=(15,10))

		self.lines=[]
		names=[r"$\theta_x^f$",r"$\theta_y^f$",r"$\theta_z^f$","$s_x^f$","$s_y^f$","$s_z^f$",
			r"$\theta_x^s$",r"$\theta_y^s$",r"$\theta_z^s$","$s_x^s$","$s_y^s$","$s_z^s$"]
		for i in range(12):
			(line,) = ax[0,0].plot([], [], label=names[i])
			self.lines.append(line)

		self.tabfun=[]
		(self.line_fun,)=ax[1,0].plot([], [], label=names[i])

		# Labels dynamiques
		self.labels = []
		for i in range(12):
			txt = ax[0,0].text(0, 0, names[i], color=self.lines[i].get_color(), fontsize=8)
			self.labels.append(txt)


		self.scatter_front_modulus = ax[0,1].scatter([], [],alpha=0.1)
		self.scatter_sagit_modulus = ax[0,2].scatter([], [],alpha=0.1)

		self.xa_front_shape=self.sampler_front.grid_xa.shape
		self.xa_sagit_shape=self.sampler_sagit.grid_xa.shape


		self.im_front_xa=ax[1,1].imshow(np.zeros(self.xa_front_shape),cmap='Reds',alpha=0.5)
		self.im_front_tof=ax[1,1].imshow(np.zeros(self.xa_front_shape),cmap='Blues',alpha=0.5)

		self.im_sagit_xa=ax[1,2].imshow(np.zeros(self.xa_sagit_shape),cmap='Reds',alpha=0.5)
		self.im_sagit_tof=ax[1,2].imshow(np.zeros(self.xa_sagit_shape),cmap='Blues',alpha=0.5)

		#self.scatter_front_points_xa= ax[1,1].scatter([], [],color='red',marker='.',s=s)
		#self.scatter_front_points_tof= ax[1,1].scatter([], [],color='blue',marker='.',s=s)

		#self.scatter_sagit_points_xa= ax[1,2].scatter([], [],color='red',marker='.',s=s)
		#self.scatter_sagit_points_tof= ax[1,2].scatter([], [],color='blue',marker='.',s=s)

		self.fig=fig
		self.ax=ax

	def update_monitor(self,intermediate_result):

		x=intermediate_result.x
		fun=intermediate_result.fun


		y=self.x0+self.components@x

		sample_points_front,sample_scalar_xa_front,sample_scalar_tof_front = self.sampler_front.sample()
		sample_points_sagit,sample_scalar_xa_sagit,sample_scalar_tof_sagit = self.sampler_sagit.sample()

		self.scatter_front_modulus.set_offsets(list(zip(sample_scalar_xa_front, sample_scalar_tof_front)))		
		self.scatter_sagit_modulus.set_offsets(list(zip(sample_scalar_xa_sagit, sample_scalar_tof_sagit)))

		self.im_front_xa.set_data(np.reshape(sample_scalar_xa_front,self.xa_front_shape))
		self.im_front_tof.set_data(np.reshape(sample_scalar_tof_front,self.xa_front_shape))

		self.im_front_xa.set_clim(vmin=0, vmax=np.percentile(sample_scalar_xa_front,99))
		self.im_front_tof.set_clim(vmin=0, vmax=np.percentile(sample_scalar_tof_front,99))

		self.im_sagit_xa.set_data(np.reshape(sample_scalar_xa_sagit,self.xa_sagit_shape))
		self.im_sagit_tof.set_data(np.reshape(sample_scalar_tof_sagit,self.xa_sagit_shape))

		self.im_sagit_xa.set_clim(vmin=0, vmax=np.percentile(sample_scalar_xa_sagit,99))
		self.im_sagit_tof.set_clim(vmin=0, vmax=np.percentile(sample_scalar_tof_sagit,99))


		self.taby.append(y)
		self.tabfun.append(fun)
		self.tabit.append(self.tabit[-1]+1)
		
		for i in range(12):
			self.lines[i].set_xdata(self.tabit)
			self.lines[i].set_ydata(np.array(self.taby)[:,i])
			self.labels[i].set_position((self.tabit[-1], self.taby[-1][i]))

		self.line_fun.set_xdata(self.tabit[1:])
		self.line_fun.set_ydata(self.tabfun)


		self.ax[0,0].relim()             # Recalcule les limites
		self.ax[0,0].autoscale_view()
		self.ax[1,0].relim()             # Recalcule les limites
		self.ax[1,0].autoscale_view()

		max_front_tof=sample_scalar_tof_front.max()
		max_front_xa=sample_scalar_xa_front.max()

		max_sagit_tof=sample_scalar_tof_sagit.max()
		max_sagit_xa=sample_scalar_xa_sagit.max()


		self.ax[0,1].set_xlim(0,max_front_xa)           
		self.ax[0,1].set_ylim(0,max_front_tof)           
 
		self.ax[0,2].set_xlim(0,max_sagit_xa)           
		self.ax[0,2].set_ylim(0,max_sagit_tof)    

		self.fig.savefig(self.tmp_png,format="png", dpi=120)
		os.replace(self.tmp_png, self.out_png)		

def appendJson(json_file,entry_name,data_to_keep):

	if os.path.exists(json_file):
		with open(json_file, "r", encoding="utf-8") as f:
			data = json.load(f)
	else:
		data={}
	
	# Met à jour si existe, sinon ajoute
	data[entry_name] = data_to_keep

	# Sauvegarder
	with open(json_file, "w", encoding="utf-8") as f:
		json.dump(data, f, indent=4, ensure_ascii=False)


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






