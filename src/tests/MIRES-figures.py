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

from sklearn.neighbors import BallTree


import numpy as np
import pyvista as pv

import matplotlib.pyplot as plt

import time


def load_data():

	path_xa_front='../../data/Patient_33_XA.nii.gz'
	grid_xa_front=SpatialGridPerspective.fromNifti(path_xa_front)

	for d in grid_xa_front.dat:
		vmax=np.percentile(grid_xa_front.dat[d],99)
		vmin=0.3*np.percentile(grid_xa_front.dat[d],99)		
		grid_xa_front.dat[d]=np.clip(grid_xa_front.dat[d],vmin,vmax)-vmin

	path_xa_sagit='../../data/Patient_33_XA_2.nii.gz'
	grid_xa_sagit=SpatialGridPerspective.fromNifti(path_xa_sagit)

	for d in grid_xa_sagit.dat:
		vmax=np.percentile(grid_xa_sagit.dat[d],99)
		vmin=0.3*np.percentile(grid_xa_sagit.dat[d],99)		
		grid_xa_sagit.dat[d]=np.clip(grid_xa_sagit.dat[d],vmin,vmax)-vmin

	path_tof='../../data/Patient_33_TOF.nii.gz'
	grid_tof=SpatialGridAffine.fromNifti(path_tof)

	path_tof_segm='../../data/Patient_33_TOF_segm.nii.gz'
	grid_tof_segm=SpatialGridAffine.fromNifti(path_tof_segm)

	grid_tof.dat['1']=grid_tof.dat['0']*((grid_tof_segm.dat['0']==2)+(grid_tof_segm.dat['0']==4)>0)
	grid_tof.dat['2']=grid_tof.dat['0']*(grid_tof_segm.dat['0']>0)
	
	return grid_xa_front,grid_xa_sagit, grid_tof



def show_all(grid_xa_front,grid_xa_sagit, grid_tof):

	idx=12

	cam_front=Camera.fromPointClouds(grid_xa_front.center,-grid_xa_front.affine[:3,0],grid_xa_front.affine[:3,1],X_to_listpoints(grid_xa_front.getCoordinates()))
	cam_sagit=Camera.fromPointClouds(grid_xa_sagit.center,-grid_xa_sagit.affine[:3,0],grid_xa_sagit.affine[:3,1],X_to_listpoints(grid_xa_sagit.getCoordinates()))

	
	grid_tof_pv=grid_tof.toPyvista()
	grid_xa_front_pv=grid_xa_front.toPyvista()
	grid_xa_sagit_pv=grid_xa_sagit.toPyvista()
	cam_front_pv=cam_front.toPyvista()
	cam_sagit_pv=cam_sagit.toPyvista()


	p = pv.Plotter(window_size=[2000, 2000])

	p.add_mesh(cam_front_pv,line_width=5,color='darkred',opacity=0.5)
	p.add_mesh(cam_sagit_pv,line_width=5,color='darkred',opacity=0.5)
	p.add_mesh(grid_xa_front_pv,cmap='Reds',opacity='linear',scalars=str(idx),show_scalar_bar=False)
	p.add_mesh(grid_xa_sagit_pv,cmap='Reds',opacity='linear',scalars=str(idx),show_scalar_bar=False)
	#p.add_mesh(grid_tof_pv.contour([50],grid_tof_pv.point_data['1']), color="blue",copy_mesh=True)	
	#p.add_mesh(grid_tof_pv.contour([50],grid_tof_pv.point_data['2']), color="blue",copy_mesh=True,opacity=0.1)	
	p.show()





if __name__=='__main__':

	grid_xa_front,grid_xa_sagit, grid_tof=load_data()
	show_all(grid_xa_front,grid_xa_sagit, grid_tof)


	