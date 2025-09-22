import sys
sys.path.append('../')
from dactim_angio.grid import SpatialGrid
from dactim_angio.grid import SpatialGridAffine
from dactim_angio.grid import SpatialGrid_XA

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
	path_xa_front_metadata='../../data/Patient_33_metadata.json'

	grid_xa_front=SpatialGrid_XA.fromNifti(path_xa_front,path_xa_front_metadata)

	for d in grid_xa_front.dat:
		vmax=np.percentile(grid_xa_front.dat[d],99)
		vmin=0.3*np.percentile(grid_xa_front.dat[d],99)		
		grid_xa_front.dat[d]=np.clip(grid_xa_front.dat[d],vmin,vmax)-vmin

	path_xa_sagit='../../data/Patient_33_XA_2.nii.gz'
	path_xa_sagit_metadata='../../data/Patient_33_metadata2.json'

	grid_xa_sagit=SpatialGrid_XA.fromNifti(path_xa_sagit,path_xa_sagit_metadata)

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


def load_data_bin():

	
	path_tof='../../data/Patient_33_TOF.nii.gz'
	grid_tof=SpatialGridAffine.fromNifti(path_tof)

	path_tof_segm='../../data/Patient_33_TOF_segm.nii.gz'
	grid_tof_segm=SpatialGridAffine.fromNifti(path_tof_segm)

	grid_tof.dat['1']=grid_tof.dat['0']*((grid_tof_segm.dat['0']==2)+(grid_tof_segm.dat['0']==4)>0)
	grid_tof.dat['2']=grid_tof.dat['0']*(grid_tof_segm.dat['0']>0)
	

	affine_front=np.zeros((4,3))
	affine_front[:,0]=grid_tof.affine[:,0]
	affine_front[:,1]=grid_tof.affine[:,2]
	affine_front[:,2]=grid_tof.affine[:,3]

	normal=np.cross(affine_front[:3,0],affine_front[:3,1])
	normal=normal/np.linalg.norm(normal)
	affine_front[:3,2]=affine_front[:3,2]+10*normal

	grid_xa_front=SpatialGridAffine((grid_tof.shape[0],grid_tof.shape[2]),affine_front,{12:np.sum(grid_tof.dat['2'],axis=1)})
	

	affine_sagit=np.zeros((4,3))
	affine_sagit[:,0]=grid_tof.affine[:,1]
	affine_sagit[:,1]=grid_tof.affine[:,2]
	affine_sagit[:,2]=grid_tof.affine[:,3]

	normal=np.cross(affine_sagit[:3,0],affine_sagit[:3,1])
	normal=normal/np.linalg.norm(normal)
	affine_sagit[:3,2]=affine_sagit[:3,2]-10*normal

	grid_xa_sagit=SpatialGridAffine((grid_tof.shape[1],grid_tof.shape[2]),affine_sagit,{12:np.sum(grid_tof.dat['2'],axis=0)})


	return grid_xa_front,grid_xa_sagit, grid_tof


def define_bounds_TOF(grid):
	X=grid.getCoordinates()

	p1=[X[0][0,0,0],X[1][0,0,0],X[2][0,0,0]]
	p2=[X[0][-1,0,0],X[1][-1,0,0],X[2][-1,0,0]]
	p3=[X[0][0,-1,0],X[1][0,-1,0],X[2][0,-1,0]]
	p4=[X[0][0,0,-1],X[1][0,0,-1],X[2][0,0,-1]]
	p5=[X[0][-1,-1,0],X[1][-1,-1,0],X[2][-1,-1,0]]
	p6=[X[0][0,-1,-1],X[1][0,-1,-1],X[2][0,-1,-1]]
	p7=[X[0][-1,0,-1],X[1][-1,0,-1],X[2][-1,0,-1]]
	p8=[X[0][-1,-1,-1],X[1][-1,-1,-1],X[2][-1,-1,-1]]

	arr=[]
	arr.append(pv.Line(p1,p2))
	arr.append(pv.Line(p2,p5))
	arr.append(pv.Line(p5,p3))
	arr.append(pv.Line(p3,p1))


	arr.append(pv.Line(p4,p6))
	arr.append(pv.Line(p6,p8))
	arr.append(pv.Line(p8,p7))
	arr.append(pv.Line(p7,p4))

	arr.append(pv.Line(p1,p4))
	arr.append(pv.Line(p2,p7))
	arr.append(pv.Line(p5,p8))
	arr.append(pv.Line(p3,p6))
	
	return pv.MultiBlock(arr)


def define_bounds_XA(grid):
	X=grid.getCoordinates()

	p1=[X[0][0,0],X[1][0,0],X[2][0,0]]
	p2=[X[0][0,-1],X[1][0,-1],X[2][0,-1]]
	p4=[X[0][-1,0],X[1][-1,0],X[2][-1,0]]
	p3=[X[0][-1,-1],X[1][-1,-1],X[2][-1,-1]]
	
	

	arr=[]
	arr.append(pv.Line(p1,p2))
	arr.append(pv.Line(p2,p3))
	arr.append(pv.Line(p3,p4))
	arr.append(pv.Line(p4,p1))
	
	return pv.MultiBlock(arr)

def show_all(grid_xa_front,grid_xa_sagit, grid_tof):

	idx=12

	
	
	grid_tof_pv=grid_tof.toPyvista()
	grid_xa_front_pv=grid_xa_front.toPyvista()
	grid_xa_sagit_pv=grid_xa_sagit.toPyvista()
	

	p = pv.Plotter(window_size=[2000, 2000])
	pv.global_theme.transparent_background = True

	max_front=np.percentile(grid_xa_front.dat[12],99)
	max_sagit=np.percentile(grid_xa_sagit.dat[12],99)

	p.add_mesh(grid_xa_front_pv,cmap='Reds',opacity='linear',scalars=str(idx),clim=[0,max_front],show_scalar_bar=False)
	p.add_mesh(grid_xa_sagit_pv,cmap='Reds',opacity='linear',scalars=str(idx),clim=[0,max_sagit],show_scalar_bar=False)
	#p.add_mesh(grid_tof_pv.contour([50],grid_tof_pv.point_data['1']), color="blue",copy_mesh=True)	
	p.add_mesh(grid_tof_pv.contour([50],grid_tof_pv.point_data['2']), color="blue",copy_mesh=True,opacity=1)	
	p.add_mesh(define_bounds_TOF(grid_tof),color="blue",line_width=5)	
	p.add_mesh(define_bounds_XA(grid_xa_front),color="red",line_width=5)	
	p.add_mesh(define_bounds_XA(grid_xa_sagit),color="red",line_width=5)	
	

	p.show()

	p.screenshot("bin.png")



def fig_3_bin():
	import numpy as np
	import matplotlib.pyplot as plt
	from matplotlib.gridspec import GridSpec
	from tensorflow.keras.datasets import mnist  # pip install tensorflow
	# --- Charger MNIST et extraire un "3"
	(x_train, y_train), _ = mnist.load_data()
	img = x_train[y_train == 3][0].astype(float)  # 28x28

	# --- Sommes
	row_sums = img.sum(axis=1)  # 28
	col_sums = img.sum(axis=0)  # 28

	# --- Figure avec 2x2 GridSpec (zone droite et bas pour les insets)
	fig = plt.figure(figsize=(6, 6), constrained_layout=False)
	gs = GridSpec(2, 2, figure=fig, width_ratios=[4, 1.2], height_ratios=[4, 1.2],
		wspace=0.05, hspace=0.05)
	
	ax_main   = fig.add_subplot(gs[0, 0])
	ax_right  = fig.add_subplot(gs[0, 1], sharey=ax_main)
	ax_bottom = fig.add_subplot(gs[1, 0], sharex=ax_main)

	# --- Image principal
	ax_main.imshow(img, cmap='Blues', origin='upper')
	ax_main.set_xticks([]); ax_main.set_yticks([])
	#for spine in ax_main.spines.values(): spine.set_visible(False)

	# --- Inset du bas : somme des lignes (1 courbe sur x = colonnes)
	ax_bottom.plot(np.arange(img.shape[1]), row_sums,color='red')
	ax_bottom.set_xlim(-0.5, img.shape[1]-0.5)
	#for spine in ax_bottom.spines.values(): spine.set_visible(False)

	# --- Inset de droite : somme des colonnes (tracée à 90°)
	# On trace col_sums en fonction de y (index de ligne) pour avoir une courbe "verticale"
	ax_right.plot(col_sums, np.arange(img.shape[0]),color='red')
	ax_right.set_ylim(img.shape[0]-0.5, -0.5)  # pour aligner le haut avec le haut de l'image
	#for spine in ax_right.spines.values(): spine.set_visible(False)

	plt.show()




if __name__=='__main__':

	grid_xa_front,grid_xa_sagit, grid_tof=load_data()
	grid_xa_front,grid_xa_sagit, grid_tof=load_data_bin()

	show_all(grid_xa_front,grid_xa_sagit, grid_tof)

	fig_3_bin()

	