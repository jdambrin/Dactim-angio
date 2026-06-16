from dactim_angio.metrics import opposite_correlation
from dactim_angio.grid import SpatialGridAffine
from dactim_angio.grid import SpatialGrid_XA
from scipy.optimize import minimize
import numpy as np
import time
from dactim_angio.sampler import Sampler_Simpler
from preproc import load_data_raw,cleanDSA,maskBG
from preproc import append_all_new
import json
import os

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from collections import OrderedDict

datapath='/Users/juliendambrine/Codes/ClementT/data_processed/PIMISUTT/'
path_xa_front=datapath+'dsa_raw_'+'front'+'_nifti_files/'
path_xa_metadata_front=datapath+'dsa_raw_'+'front'+'_nifti_files/'
		
path_xa_sagit=datapath+'dsa_raw_'+'sagit'+'_nifti_files/'
path_xa_metadata_sagit=datapath+'dsa_raw_'+'sagit'+'_nifti_files/'
		

path_tof=datapath+'tof_nifti_files/'
path_tof_segm=datapath+'topcow_to_full_segmentation/'


times={}

do_not_subtract=['Patient_45','Patient_46','Patient_47','Patient_48']

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

	@measure_time
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

@measure_time
def register(grid_xa_front,grid_xa_sagit, grid_tof,components,x0,metric):

	field=('all','all')

	dim = components.shape[1]

	if(metric[2]=='minimize'):
		sign=1
	elif(metric[2]=='maximize'):
		sign=-1
	else:
		raise Exception('Unknown optimization mode')

	@measure_time
	def func_to_minimize(x,*args):

		sampler_front=args[0]
		sampler_sagit=args[1]
		y=x0+components@x

		sampler_front.update_tof_points(y[0],y[1],y[2],y[3],y[4],y[5])
		sampler_sagit.update_tof_points(y[6],y[7],y[8],y[9],y[10],y[11])

		sample_points_front,sample_scalar_xa_front,sample_scalar_tof_front=sampler_front.sample()
		sample_points_sagit,sample_scalar_xa_sagit,sample_scalar_tof_sagit=sampler_sagit.sample()

		f = sign*metric[0](sample_scalar_xa_front,sample_scalar_tof_front,**(metric[1]))[0]
		f+= sign*metric[0](sample_scalar_xa_sagit,sample_scalar_tof_sagit,**(metric[1]))[0]

		return f

	@measure_time
	def Jac_func_to_minimize(x,*args):
		sampler_front=args[0]
		sampler_sagit=args[1]
		y=x0+components@x
		sampler_front.update_tof_points(y[0],y[1],y[2],y[3],y[4],y[5])
		sampler_sagit.update_tof_points(y[6],y[7],y[8],y[9],y[10],y[11])
		sample_points_front,sample_scalar_xa_front,sample_scalar_tof_front=sampler_front.sample()
		sample_points_sagit,sample_scalar_xa_sagit,sample_scalar_tof_sagit=sampler_sagit.sample()


		euler_gradient_front=sampler_front.euler_gradient_sample()
		euler_gradient_sagit=sampler_sagit.euler_gradient_sample()

		corr_front,deriv_corr_x_front,deriv_corr_y_front=metric[0](sample_scalar_xa_front,sample_scalar_tof_front,**(metric[1]))
		corr_sagit,deriv_corr_x_sagit,deriv_corr_y_sagit=metric[0](sample_scalar_xa_sagit,sample_scalar_tof_sagit,**(metric[1]))

		corr_front*=sign
		deriv_corr_x_front*=sign
		deriv_corr_y_front*=sign

		corr_sagit*=sign
		deriv_corr_x_sagit*=sign
		deriv_corr_y_sagit*=sign

		deriv_samples_wrt_xa_front,deriv_samples_wrt_tof_front=np.real(euler_gradient_front)
		deriv_samples_wrt_xa_sagit,deriv_samples_wrt_tof_sagit=np.real(euler_gradient_sagit)

		gradf = np.dot([np.dot(deriv_corr_x_front,deriv_samples_wrt_xa_front[i])+np.dot(deriv_corr_y_front,deriv_samples_wrt_tof_front[i]) for i in range(6)],components[[0,1,2,3,4,5],:]) + np.dot([np.dot(deriv_corr_x_sagit,deriv_samples_wrt_xa_sagit[i])+np.dot(deriv_corr_y_sagit,deriv_samples_wrt_tof_sagit[i]) for i in range(6)],components[[6,7,8,9,10,11],:])

		
		return gradf



	mySampler_front=Sampler_Simpler(grid_xa_front,grid_tof,field)
	mySampler_front.update_tof_points(x0[0],x0[1],x0[2],x0[3],x0[4],x0[5])

	mySampler_sagit=Sampler_Simpler(grid_xa_sagit,grid_tof,field)
	mySampler_sagit.update_tof_points(x0[6],x0[7],x0[8],x0[9],x0[10],x0[11])

	sample_points_front,sample_scalar_xa_front,sample_scalar_tof_front=mySampler_front.sample()


	myMonitor = Monitor(mySampler_front,mySampler_sagit,x0,components,'Monitoring/live.png')

	def cb(intermediate_result):
		myMonitor.update_monitor(intermediate_result)

	x=x0
	sample_points_front,sample_scalar_xa_front,sample_scalar_tof_front=mySampler_front.sample()
	sample_points_sagit,sample_scalar_xa_sagit,sample_scalar_tof_sagit=mySampler_sagit.sample()

	y0=np.zeros(dim)
	print('minimize')

	res=minimize(func_to_minimize,
		y0,
		args=(mySampler_front,mySampler_sagit),
		method='L-BFGS-B',
		jac=Jac_func_to_minimize, 
		callback= cb,
		)
	x=x0+components@res.x
	print(res)

	plt.close(myMonitor.fig)

	return x


def multiScaleRegister(grid_xa_front, grid_xa_sagit, grid_tof,levels,components,metrics,x0=[0,0,0,0,0,0,0,0,0,0,0,0]):
	
	pyramid_tof=SpatialGridAffine.build_pyramid(grid_tof)
	pyramid_xa_front=SpatialGrid_XA.build_pyramid(grid_xa_front)
	pyramid_xa_sagit=SpatialGrid_XA.build_pyramid(grid_xa_sagit)

	#for i in range(len(sigma_add)):

	for i in range(len(levels)):

		grid_xa_rescaled_front=pyramid_xa_front[levels[i][0]]
		grid_xa_rescaled_sagit=pyramid_xa_sagit[levels[i][0]]
		grid_tof_rescaled=pyramid_tof[levels[i][1]]

		x=register(grid_xa_rescaled_front, grid_xa_rescaled_sagit, grid_tof_rescaled,components[i],x0,metrics[i])	
		x0=x

		print(times)
		for key in times:
			times[key]=0

	return x


if __name__=='__main__':

	out_json_file='experiment3.json'

	with open("selected_arterial_frames.json", "r", encoding="utf-8") as f:
		in_dict = json.load(f, object_pairs_hook=OrderedDict)


	patients=list(in_dict.keys())
	print(patients)

	#for p in patients[10:]:
	#for p in list(in_dict.keys())[3:]:
	for p in ['Patient_52']:
	#for p in patients:
		print(p)

		patient_name=p

		file_xa_front=path_xa_front+patient_name+'.nii.gz'
		file_xa_metadata_front=path_xa_metadata_front+patient_name+'_metadata.json'
		
		file_xa_sagit=path_xa_sagit+patient_name+'.nii.gz'
		file_xa_metadata_sagit=path_xa_metadata_sagit+patient_name+'_metadata.json'
		
		file_tof=path_tof+patient_name+'.nii.gz'
		file_tof_segm=path_tof_segm+patient_name+'_CoW_seg.nii.gz'
		
		side=in_dict[patient_name]['side']

		xa_frame_mca_front=in_dict[patient_name]['front']['mca_frame']
		xa_frame_mca_sagit=in_dict[patient_name]['sagit']['mca_frame']
		
		grid_xa_front,grid_xa_sagit,grid_tof, grid_tof_segm =load_data_raw(file_xa_front,file_xa_metadata_front,file_xa_sagit,file_xa_metadata_sagit,file_tof,file_tof_segm)

		if p in do_not_subtract:
			sub=False
		else:
			sub=True

		grid_xa_front=cleanDSA(grid_xa_front,border_thickness=70,subtract=sub,first_index=0)
		grid_xa_front_mask=maskBG(grid_xa_front)

		grid_xa_sagit=cleanDSA(grid_xa_sagit,border_thickness=70,subtract=sub,first_index=0)
		grid_xa_sagit_mask=maskBG(grid_xa_sagit)

		grid_xa_front, grid_xa_sagit,grid_tof=append_all_new(grid_xa_front, grid_xa_front_mask,grid_xa_sagit, grid_xa_sagit_mask, grid_tof, grid_tof_segm, side, xa_frame_mca_front, xa_frame_mca_sagit)


		Nframes_front=len(grid_xa_front.dat)
		Nframes_sagit=len(grid_xa_sagit.dat)


		v1_front=grid_xa_front.affine[:3,0].copy()
		v1_front/=np.linalg.norm(v1_front)
		v2_front=grid_xa_front.affine[:3,1].copy()
		v2_front/=np.linalg.norm(v2_front)

		vv1_front=np.zeros(12)
		vv2_front=np.zeros(12)
		vv1_front[3:6]=v1_front
		vv2_front[3:6]=v2_front
		

		v1_sagit=grid_xa_sagit.affine[:3,0].copy()
		v1_sagit/=np.linalg.norm(v1_sagit)
		v2_sagit=grid_xa_sagit.affine[:3,1].copy()
		v2_sagit/=np.linalg.norm(v2_sagit)

		vv1_sagit=np.zeros(12)
		vv2_sagit=np.zeros(12)
		vv1_sagit[9:12]=v1_sagit
		vv2_sagit[9:12]=v2_sagit

		
		x0=np.zeros(12)

		all_components=np.eye(12)

		linked_thetas_tangent=np.array([
			[1,0,0,0,0,0,1,0,0,0,0,0],
			[0,1,0,0,0,0,0,1,0,0,0,0],
			[0,0,1,0,0,0,0,0,1,0,0,0],
			vv1_front.tolist(),
			vv2_front.tolist(),
			vv1_sagit.tolist(),
			vv2_sagit.tolist()
			]).T

		free_thetas_tangent=np.array([
			[1,0,0,0,0,0,0,0,0,0,0,0],
			[0,1,0,0,0,0,0,0,0,0,0,0],
			[0,0,1,0,0,0,0,0,0,0,0,0],
			[0,0,0,0,0,0,1,0,0,0,0,0],
			[0,0,0,0,0,0,0,1,0,0,0,0],
			[0,0,0,0,0,0,0,0,1,0,0,0],
			vv1_front.tolist(),
			vv2_front.tolist(),
			vv1_sagit.tolist(),
			vv2_sagit.tolist()
			]).T

		just_shifts=np.array([
			vv1_front.tolist(),
			vv2_front.tolist(),
			vv1_sagit.tolist(),
			vv2_sagit.tolist()
			]).T

		just_shifts_linked=np.array([
			[0,0,0,1,0,0,0,0,0,1,0,0],
			[0,0,0,0,1,0,0,0,0,0,1,0],
			[0,0,0,0,0,1,0,0,0,0,0,1]
			]).T

		linked_thetas=np.array([
			[1,0,0,0,0,0,1,0,0,0,0,0],
			[0,1,0,0,0,0,0,1,0,0,0,0],
			[0,0,1,0,0,0,0,0,1,0,0,0],
			[0,0,0,1,0,0,0,0,0,0,0,0],
			[0,0,0,0,1,0,0,0,0,0,0,0],
			[0,0,0,0,0,1,0,0,0,0,0,0],
			[0,0,0,0,0,0,0,0,0,1,0,0],
			[0,0,0,0,0,0,0,0,0,0,1,0],
			[0,0,0,0,0,0,0,0,0,0,0,1]
			]).T

		just_shifts=np.array([
			vv1_front.tolist(),
			vv2_front.tolist(),
			vv1_sagit.tolist(),
			vv2_sagit.tolist()
			]).T


		levels=[
			(6,5),
			(5,4),
			(5,4),
			(5,3),
			(4,2),
			(3,1),
			(2,0),
			(2,0),
			]


		components=[
			just_shifts_linked,
			just_shifts[:,:2],
			just_shifts[:,2:],
			linked_thetas_tangent,
			linked_thetas,
			linked_thetas,
			linked_thetas,
			all_components,
			]

		metrics=[
			(opposite_correlation,{},'minimize'),
			(opposite_correlation,{},'minimize'),
			(opposite_correlation,{},'minimize'),
			(opposite_correlation,{},'minimize'),
			(opposite_correlation,{},'minimize'),
			(opposite_correlation,{},'minimize'),
			(opposite_correlation,{},'minimize'),
			(opposite_correlation,{},'minimize'),
			]
		


		x=multiScaleRegister(grid_xa_front,grid_xa_sagit, grid_tof,levels,components,metrics,x0)
		
		data_to_keep={
			'side':side,
			'file_tof':file_tof,
			'file_tof_segm':file_tof_segm,
			'exec_time':times,
			'front':{
				'theta_x':float(x[0]),'theta_y':float(x[1]),'theta_z':float(x[2]),
				'shift_x':float(x[3]),'shift_y':float(x[4]),'shift_z':float(x[5]),
				'file_xa':file_xa_front,
				'file_xa_metadata':file_xa_metadata_front
				},
			'sagit':{
				'theta_x':float(x[6]),'theta_y':float(x[7]),'theta_z':float(x[8]),
				'shift_x':float(x[9]),'shift_y':float(x[10]),'shift_z':float(x[11]),
				'file_xa':file_xa_sagit,
				'file_xa_metadata':file_xa_metadata_sagit
				}
			}
		
		appendJson(out_json_file,p,data_to_keep)

	print(times)


