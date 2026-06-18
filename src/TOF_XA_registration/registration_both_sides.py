from dactim_angio.metrics import opposite_correlation
from dactim_angio.spatial.grid import SpatialGridAffine
from dactim_angio.spatial.grid import SpatialGrid_XA

from scipy.optimize import minimize
import numpy as np
import time
from utils import Sampler_Simpler, Monitor, appendJson
from preproc import load_data_raw,cleanDSA,maskBG,select_all
import json

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from collections import OrderedDict


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

#do_not_subtract=['Patient_45','Patient_46','Patient_47','Patient_48']

@measure_time
def register(grid_xa_front,grid_xa_sagit, grid_tof,components,x0,metric):

	field=('selected','selected')

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


	myMonitor = Monitor(mySampler_front,mySampler_sagit,x0,components,'_monitoring/live.png')

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

	out_json_file='../../results/output.json'

	with open("input_files_setup.json", "r", encoding="utf-8") as f:
		in_dict = json.load(f, object_pairs_hook=OrderedDict)


	patients=list(in_dict.keys())
	print(patients)

	for p in patients:

		print(p)

		patient_name=p
		
		file_xa_front =          in_dict[patient_name]['front']['nifti_file']
		file_xa_metadata_front = in_dict[patient_name]['front']['metadata']
		
		file_xa_sagit =          in_dict[patient_name]['sagit']['nifti_file']
		file_xa_metadata_sagit = in_dict[patient_name]['sagit']['metadata']
		
		
		file_tof=in_dict[patient_name]['tof_nifti_file']
		file_tof_segm=in_dict[patient_name]['tof_segm_nifti_file']

		side=in_dict[patient_name]['side']

		xa_frame_mca_front=in_dict[patient_name]['front']['mca_frame']
		xa_frame_mca_sagit=in_dict[patient_name]['sagit']['mca_frame']

		cleanDSA_kwargs_front=in_dict[patient_name]['front']['cleanDSA_args']
		cleanDSA_kwargs_sagit=in_dict[patient_name]['sagit']['cleanDSA_args']

		grid_xa_front,grid_xa_sagit,grid_tof, grid_tof_segm =load_data_raw(file_xa_front,file_xa_metadata_front,file_xa_sagit,file_xa_metadata_sagit,file_tof,file_tof_segm)


		grid_xa_front=cleanDSA(grid_xa_front,**cleanDSA_kwargs_front)
		grid_xa_front_mask=maskBG(grid_xa_front)

		grid_xa_sagit=cleanDSA(grid_xa_sagit,**cleanDSA_kwargs_sagit)
		grid_xa_sagit_mask=maskBG(grid_xa_sagit)

		grid_xa_front, grid_xa_sagit,grid_tof=select_all(grid_xa_front, grid_xa_front_mask,grid_xa_sagit, grid_xa_sagit_mask, grid_tof, grid_tof_segm, side, xa_frame_mca_front, xa_frame_mca_sagit)


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


