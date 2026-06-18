from dactim_angio.spatial.grid import SpatialGridAffine
from dactim_angio.spatial.grid import SpatialGrid_XA
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage import measure, morphology, data
from skimage.morphology import dilation, square
import skfmm

import matplotlib.pyplot as plt

dict_tof_seg={'left_ica':1,'left_mca':2,'right_ica':3,'right_mca':4,'posterior':5,'anterior':6}


def load_data_raw(path_xa_front,path_xa_front_metadata, path_xa_sagit,path_xa_sagit_metadata,path_tof, path_tof_segm):

    grid_xa_front = SpatialGrid_XA.fromNifti(path_xa_front, path_xa_front_metadata)
    grid_xa_sagit = SpatialGrid_XA.fromNifti(path_xa_sagit, path_xa_sagit_metadata)
    grid_tof = SpatialGridAffine.fromNifti(path_tof)
    grid_tof_segm = SpatialGridAffine.fromNifti(path_tof_segm)

    return grid_xa_front,grid_xa_sagit, grid_tof, grid_tof_segm



def cleanDSA(grid_xa,border_thickness,subtract=True,first_index=0):
    band=border_thickness
    for d in grid_xa.dat:
        
        background_val=np.percentile(grid_xa.dat[d][band:-band,band:-band],90)
        grid_xa.dat[d]=background_val-grid_xa.dat[d]

    if(subtract):
        reference=grid_xa.dat[first_index].copy()
        for d in grid_xa.dat:
            grid_xa.dat[d]=grid_xa.dat[d]-reference

    for d in grid_xa.dat:
        tmp=np.zeros(grid_xa.shape)
        tmp[band:-band,band:-band]=grid_xa.dat[d][band:-band,band:-band]
        grid_xa.dat[d]=tmp
        grid_xa.dat[d]=np.clip(grid_xa.dat[d],0,None)

    return grid_xa

def maskBG(grid_xa):

    grid_xa_mask=grid_xa.copy()
    N1,N2=grid_xa.shape
    N3=len(grid_xa.dat)
    seq=np.zeros((N1,N2,N3))

    for i in range(N3):
        seq[:,:,i]=grid_xa.dat[i]
        
    binary=seq>0.2*seq.max()
    mask=0*seq
    N3=seq.shape[2]
    
    for i in range(N3):
        label_image = measure.label(binary[:,:,i])
        cleaned = morphology.remove_small_objects(label_image, min_size=2000)
        cleaned_binary = cleaned > 0
        dilated_binary = dilation(cleaned_binary, square(5)) 
        mask[:,:,i]=dilated_binary
        grid_xa_mask.dat[i]=dilated_binary
        
    return grid_xa_mask


        
def select_all(grid_xa_front, grid_xa_front_mask, grid_xa_sagit, grid_xa_sagit_mask, grid_tof, grid_tof_segm, side, xa_frame_mca_front, xa_frame_mca_sagit):

    grid_xa_front.dat["selected"]=grid_xa_front.dat[xa_frame_mca_front]*grid_xa_front_mask.dat[xa_frame_mca_front]
    grid_xa_sagit.dat["selected"]=grid_xa_sagit.dat[xa_frame_mca_sagit]*grid_xa_sagit_mask.dat[xa_frame_mca_sagit]

    if (side=='right'):
        grid_tof.dat["selected"] = ((grid_tof_segm.dat["0"]==dict_tof_seg['right_ica'])+(grid_tof_segm.dat["0"]==dict_tof_seg['right_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['anterior'])>0).astype('float')
    else:
        grid_tof.dat["selected"] = ((grid_tof_segm.dat["0"]==dict_tof_seg['left_ica'])+(grid_tof_segm.dat["0"]==dict_tof_seg['left_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['anterior'])>0).astype('float')


    grid_tof.dat["selected"] *= grid_tof.dat["0"]
    

    grid_tof.dat["both_sides"] = grid_tof.dat["0"]*((grid_tof_segm.dat["0"]==dict_tof_seg['right_ica'])+(grid_tof_segm.dat["0"]==dict_tof_seg['right_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['left_ica'])+(grid_tof_segm.dat["0"]==dict_tof_seg['left_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['anterior'])>0).astype('float')

    return grid_xa_front,grid_xa_sagit, grid_tof

