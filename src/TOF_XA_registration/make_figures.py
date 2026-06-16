from dactim_angio.metrics import opposite_mutual_information_kernel
from dactim_angio.metrics import opposite_correlation
from dactim_angio.metrics import singular_exclusion
from dactim_angio.metrics import mse
from dactim_angio.metrics import joint_entropy_kraskov

from dactim_angio.grid import SpatialGridAffine
from dactim_angio.grid import SpatialGrid_XA


from dactim_angio.geom import getAffine


import numpy as np

from preproc import load_data_raw,cleanDSA,maskBG
from preproc import append_all_new


from plotting import Envplotter
import json
import os

dict_tof_seg={'left_ica':1,'left_mca':2,'right_ica':3,'right_mca':4,'posterior':5,'anterior':6}

datapath='/Users/juliendambrine/Codes/ClementT/data_processed/PIMISUTT/'
path_xa_front=datapath+'dsa_raw_'+'front'+'_nifti_files/'
path_xa_metadata_front=datapath+'dsa_raw_'+'front'+'_nifti_files/'
		
path_xa_sagit=datapath+'dsa_raw_'+'sagit'+'_nifti_files/'
path_xa_metadata_sagit=datapath+'dsa_raw_'+'sagit'+'_nifti_files/'
		

path_tof=datapath+'tof_nifti_files/'
path_tof_segm=datapath+'topcow_to_full_segmentation/'

do_not_subtract=['Patient_45','Patient_46','Patient_47','Patient_48']


def makefigs():


	result='experiment3.json'

	with open(result, "r", encoding="utf-8") as f:
		data_register = json.load(f)

	with open("./selected_arterial_frames.json", "r", encoding="utf-8") as f:
		in_dict = json.load(f)

	for p in data_register:
	#for p in ['Patient_2']:

		splitted=p.split('_')

		patient_id=int(splitted[1])

		patient_name=p


		side=in_dict['Patient_'+str(patient_id)]['side']

		file_xa_front=path_xa_front+patient_name+'.nii.gz'
		file_xa_metadata_front=path_xa_metadata_front+patient_name+'_metadata.json'
		
		file_xa_sagit=path_xa_sagit+patient_name+'.nii.gz'
		file_xa_metadata_sagit=path_xa_metadata_sagit+patient_name+'_metadata.json'
		
		file_tof=path_tof+patient_name+'.nii.gz'
		file_tof_segm=path_tof_segm+patient_name+'_CoW_seg.nii.gz'
		side=in_dict[p]['side']

		xa_frame_ica_front=in_dict[p]['front']['ica_frame']
		xa_frame_mca_front=in_dict[p]['front']['mca_frame']

		xa_frame_ica_sagit=in_dict[p]['sagit']['ica_frame']
		xa_frame_mca_sagit=in_dict[p]['sagit']['mca_frame']

		grid_xa_front,grid_xa_sagit,grid_tof, grid_tof_segm =load_data_raw(file_xa_front,file_xa_metadata_front,file_xa_sagit,file_xa_metadata_sagit,file_tof,file_tof_segm)
		
		Nframes_front=len(grid_xa_front.dat)
		Nframes_sagit=len(grid_xa_sagit.dat)

		if p in do_not_subtract:
			sub=False
		else:
			sub=True

		grid_xa_front=cleanDSA(grid_xa_front,border_thickness=70,subtract=sub,first_index=0)
		grid_xa_front_mask=maskBG(grid_xa_front)

		grid_xa_sagit=cleanDSA(grid_xa_sagit,border_thickness=70,subtract=sub,first_index=0)
		grid_xa_sagit_mask=maskBG(grid_xa_sagit)

		grid_xa_front, grid_xa_sagit,grid_tof=append_all_new(grid_xa_front, grid_xa_front_mask,grid_xa_sagit, grid_xa_sagit_mask, grid_tof, grid_tof_segm, side, xa_frame_mca_front, xa_frame_mca_sagit)

		

		max_front=0
		max_sagit=0

		Nframes=np.min([Nframes_front,Nframes_sagit])

		for i in range(Nframes):
			grid_xa_front.dat[i]=np.clip(grid_xa_front.dat[i],0,None)
			grid_xa_sagit.dat[i]=np.clip(grid_xa_sagit.dat[i],0,None)
			if(grid_xa_front.dat[i].max()>max_front):
				max_front=grid_xa_front.dat[i].max()
			if(grid_xa_sagit.dat[i].max()>max_sagit):
				max_sagit=grid_xa_sagit.dat[i].max()

		theta_x_front=data_register[p]['front']['theta_x']
		theta_y_front=data_register[p]['front']['theta_y']
		theta_z_front=data_register[p]['front']['theta_z']
		shift_x_front=data_register[p]['front']['shift_x']
		shift_y_front=data_register[p]['front']['shift_y']
		shift_z_front=data_register[p]['front']['shift_z']

		correction_affine_front=getAffine(theta_x_front,theta_y_front,theta_z_front,shift_x_front,shift_y_front,shift_z_front)[0]
		
		theta_x_sagit=data_register[p]['sagit']['theta_x']
		theta_y_sagit=data_register[p]['sagit']['theta_y']
		theta_z_sagit=data_register[p]['sagit']['theta_z']
		shift_x_sagit=data_register[p]['sagit']['shift_x']
		shift_y_sagit=data_register[p]['sagit']['shift_y']
		shift_z_sagit=data_register[p]['sagit']['shift_z']

		correction_affine_sagit=getAffine(theta_x_sagit,theta_y_sagit,theta_z_sagit,shift_x_sagit,shift_y_sagit,shift_z_sagit)[0]
		

		dir_name=result.split('.')[0]

		out_path=dir_name+'/'+p+'/'
		
	
		os.makedirs(out_path, exist_ok=True)
	
		print(p)

		envplotter=Envplotter(grid_xa_front,grid_xa_sagit,grid_tof,affine_front = correction_affine_front,affine_sagit = correction_affine_sagit,max_xa_front=0.9*max_front,max_xa_sagit=0.9*max_sagit)


		for i in range(Nframes):	
			print('\t Frame ',i,'/',Nframes-1)
			out_file=out_path+'frame'+f'{i:05d}'+'.png'
			envplotter.plot_xa_tof_with_sample_points_front_sagit(out_file,str(i))



if __name__=='__main__':
	makefigs()

	


