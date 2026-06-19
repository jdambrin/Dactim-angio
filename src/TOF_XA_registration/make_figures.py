from dactim_angio.spatial.geom import getAffine

import numpy as np
from preproc import load_data_raw,cleanDSA,maskBG,select_all
import json

import os
from collections import OrderedDict

import pyvista as pv



dict_tof_seg={'left_ica':1,'left_mca':2,'right_ica':3,'right_mca':4,'posterior':5,'anterior':6}


class Envplotter:
    

    def __init__(self,grid_xa_front,grid_xa_sagit,grid_tof,affine_front = np.eye(4),affine_sagit = np.eye(4),max_xa_front=0,max_xa_sagit=0):

        self.p = pv.Plotter(off_screen=True,shape=(1, 2),window_size=[2000, 1000])

        grid_tof_plot_front=grid_tof.copy()
        grid_tof_plot_sagit=grid_tof.copy()

        grid_tof_plot_front.affine=affine_front@grid_tof_plot_front.affine
        grid_tof_plot_sagit.affine=affine_sagit@grid_tof_plot_sagit.affine

        self.grid_tof_pv_front = grid_tof_plot_front.toPyvista()
        self.grid_tof_pv_sagit = grid_tof_plot_sagit.toPyvista()
        self.grid_xa_pv_front = grid_xa_front.toPyvista()
        self.grid_xa_pv_sagit = grid_xa_sagit.toPyvista()

        self.cam1=pv.Camera()
        self.cam1.position = tuple(grid_xa_front.center)
        center1 = grid_xa_front.push(np.array([0.5 * grid_xa_front.shape[0], 0.5 * grid_xa_front.shape[1]]))
        up1 = grid_xa_front.push(np.array([0, 0.5 * grid_xa_front.shape[1]])) - center1
        dist1 = np.linalg.norm(center1-grid_xa_front.center)
        angle1 = 2 * np.arctan(np.linalg.norm(up1) / dist1) * 180 / np.pi
        self.cam1.focal_point = tuple(center1)
        self.cam1.up = tuple(np.array(up1))
        self.cam1.view_angle = angle1

        self.cam2=pv.Camera()
        self.cam2.position = tuple(grid_xa_sagit.center)
        center2 = grid_xa_sagit.push(np.array([0.5 * grid_xa_sagit.shape[0], 0.5 * grid_xa_sagit.shape[1]]))
        up2 = grid_xa_sagit.push(np.array([0, 0.5 * grid_xa_sagit.shape[1]])) - center2
        dist2 = np.linalg.norm(center2-grid_xa_sagit.center)
        angle2 = 2 * np.arctan(np.linalg.norm(up2) / dist2) * 180 / np.pi
        self.cam2.focal_point = tuple(center2)
        self.cam2.up = tuple(np.array(up2))
        self.cam2.view_angle = angle2

        self.max_xa_front=max_xa_front
        self.max_xa_sagit=max_xa_sagit




    def plot_xa_tof_with_sample_points_front_sagit(
        self,out_file,field_xa
    ):
        self.p.clear()
       

        self.p.subplot(0, 0)

        self.p.add_mesh(
            self.grid_xa_pv_front, cmap="Reds", opacity="linear", scalars=field_xa, clim=[0,0.7*self.max_xa_front], show_scalar_bar=False
        )


        self.p.add_mesh(
            self.grid_tof_pv_front.contour(
                [0.2*self.grid_tof_pv_front.point_data['selected'].max()], 
                self.grid_tof_pv_front.point_data['selected']
            ),
            color="blue",
            copy_mesh=True,
            opacity=0.2,
        )


        self.p.add_mesh(
            self.grid_tof_pv_front.contour(
                [0.2*self.grid_tof_pv_front.point_data['both_sides'].max()], 
                self.grid_tof_pv_front.point_data['both_sides']
            ),
            color="black",
            copy_mesh=True,
            opacity=0.1,
        )

        self.p.camera=self.cam1

        self.p.subplot(0, 1)

        self.p.add_mesh(
            self.grid_xa_pv_sagit, cmap="Reds", opacity="linear", scalars=field_xa, clim=[0,0.7*self.max_xa_sagit], show_scalar_bar=False
        )


        self.p.add_mesh(
            self.grid_tof_pv_sagit.contour(
                [0.2*self.grid_tof_pv_sagit.point_data['selected'].max()], 
                self.grid_tof_pv_sagit.point_data['selected']
            ),
            color="blue",
            copy_mesh=True,
            opacity=0.2,
        )

        self.p.add_mesh(
            self.grid_tof_pv_sagit.contour(
                [0.2*self.grid_tof_pv_sagit.point_data['both_sides'].max()], 
                self.grid_tof_pv_sagit.point_data['both_sides']
            ),
            color="black",
            copy_mesh=True,
            opacity=0.1,
        )

        self.p.camera=self.cam2

        #print('camera 2')
        #print(self.p.camera)

        self.p.screenshot(out_file)


if __name__=='__main__':

	with open("input_files_setup_example.json", "r", encoding="utf-8") as f:
		in_dict = json.load(f, object_pairs_hook=OrderedDict)

	with open("../../results/reg_parameters_example.json", "r", encoding="utf-8") as f:
		data_register = json.load(f, object_pairs_hook=OrderedDict)


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

		Nframes_front=len(grid_xa_front.dat)
		Nframes_sagit=len(grid_xa_sagit.dat)

		grid_xa_front=cleanDSA(grid_xa_front,**cleanDSA_kwargs_front)
		grid_xa_front_mask=maskBG(grid_xa_front)

		grid_xa_sagit=cleanDSA(grid_xa_sagit,**cleanDSA_kwargs_sagit)
		grid_xa_sagit_mask=maskBG(grid_xa_sagit)

		grid_xa_front, grid_xa_sagit,grid_tof=select_all(grid_xa_front, grid_xa_front_mask,grid_xa_sagit, grid_xa_sagit_mask, grid_tof, grid_tof_segm, side, xa_frame_mca_front, xa_frame_mca_sagit)


		

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
		

		dir_name="../../results/figures_example"

		out_path=dir_name+'/'+p+'/'
		
	
		os.makedirs(out_path, exist_ok=True)
	
		envplotter=Envplotter(grid_xa_front,grid_xa_sagit,grid_tof,affine_front = correction_affine_front,affine_sagit = correction_affine_sagit,max_xa_front=0.9*max_front,max_xa_sagit=0.9*max_sagit)


		for i in range(Nframes):	
			print('\t Frame ',i,'/',Nframes-1)
			out_file=out_path+'frame'+f'{i:05d}'+'.png'
			envplotter.plot_xa_tof_with_sample_points_front_sagit(out_file,str(i))


	


