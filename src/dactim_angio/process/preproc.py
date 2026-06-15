from dactim_angio.spatial.grid import SpatialGridAffine
from dactim_angio.spatial.grid import SpatialGrid_XA
import numpy as np
from skimage.restoration import denoise_tv_chambolle
from skimage import measure, morphology, data
from skimage.morphology import dilation, square
import skfmm

import matplotlib.pyplot as plt

dict_tof_seg={'left_ica':1,'left_mca':2,'right_ica':3,'right_mca':4,'posterior':5,'anterior':6}

def load_data(path_xa, path_xa_mask,path_xa_metadata, path_tof, path_tof_segm):

    grid_xa = SpatialGrid_XA.fromNifti(path_xa, path_xa_metadata)
    grid_xa_mask = SpatialGrid_XA.fromNifti(path_xa_mask, path_xa_metadata)
    grid_tof = SpatialGridAffine.fromNifti(path_tof)
    grid_tof_segm = SpatialGridAffine.fromNifti(path_tof_segm)

    return grid_xa, grid_xa_mask, grid_tof, grid_tof_segm


def append_ica(grid_xa, grid_xa_mask, grid_tof, grid_tof_segm, side, xa_frame_ica):

    grid_xa.dat["ica"]=grid_xa.dat[xa_frame_ica]*grid_xa_mask.dat[xa_frame_ica]
    grid_xa.dat["ica"]=np.clip(grid_xa.dat["ica"],0,None)


    if (side=='right'):
        grid_tof.dat["ica"] = (grid_tof_segm.dat["0"]==dict_tof_seg['right_ica']).astype('float')
    else:
        grid_tof.dat["ica"] = (grid_tof_segm.dat["0"]==dict_tof_seg['left_ica']).astype('float')

    grid_tof.dat["ica"] *= grid_tof.dat["0"]

    grid_tof.dat["all"] = (grid_tof_segm.dat["0"] > 0).astype("float")
    grid_tof.dat["all"] *= grid_tof.dat["0"]


    return grid_xa, grid_tof


def append_mca(grid_xa, grid_xa_mask, grid_tof, grid_tof_segm, side, xa_frame_ica,xa_frame_mca):

    grid_xa.dat["mca"]=(grid_xa.dat[xa_frame_mca]-grid_xa.dat[xa_frame_ica])*grid_xa_mask.dat[xa_frame_mca]
    grid_xa.dat["mca"]=np.clip(grid_xa.dat["mca"],0,None)


    if (side=='right'):
        grid_tof.dat["mca"] = ((grid_tof_segm.dat["0"]==dict_tof_seg['right_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['anterior'])>0).astype('float')
    else:
        grid_tof.dat["mca"] = ((grid_tof_segm.dat["0"]==dict_tof_seg['left_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['anterior'])>0).astype('float')

    grid_tof.dat["mca"] *= grid_tof.dat["0"]

    return grid_xa, grid_tof


def append_all(grid_xa, grid_xa_mask, grid_tof, grid_tof_segm, side, xa_frame_mca):

    grid_xa.dat["all"]=grid_xa.dat[xa_frame_mca]*grid_xa_mask.dat[xa_frame_mca]
    #grid_xa.dat["all"]=grid_xa.dat[xa_frame_mca]
    grid_xa.dat["all"]=np.clip(grid_xa.dat["all"],0,None)


    if (side=='right'):
        grid_tof.dat["all"] = ((grid_tof_segm.dat["0"]==dict_tof_seg['right_ica'])+(grid_tof_segm.dat["0"]==dict_tof_seg['right_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['anterior'])>0).astype('float')
    else:
        grid_tof.dat["all"] = ((grid_tof_segm.dat["0"]==dict_tof_seg['left_ica'])+(grid_tof_segm.dat["0"]==dict_tof_seg['left_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['anterior'])>0).astype('float')


    grid_tof.dat["all"] *= grid_tof.dat["0"]

    grid_tof.dat["all_both_sides"]=(grid_tof_segm.dat["0"]>0)*grid_tof.dat["0"]

    return grid_xa, grid_tof




def append_times(grid_xa, grid_xa_mask, grid_tof, grid_tof_segm, side, xa_frame_mca, Nframes):

    xa=np.array([grid_xa.dat[i] for i in range(Nframes)])
    xa_mask=np.array([grid_xa_mask.dat[i] for i in range(Nframes)])

    Tmax=xa_frame_mca+3
    phase=np.zeros(xa.shape)+0*1j

    for i in range(Tmax):
        phase[i,:,:]+=np.exp(2*np.pi*1j*i/(Tmax-1))

    tab=xa*xa_mask*phase*xa_mask[xa_frame_mca,:,:]

    u=np.mean(tab,axis=0)

    angle_flatten=np.angle(u).flatten()
    vmin=np.percentile(angle_flatten,1,weights=np.abs(u).flatten(),method='inverted_cdf')
    vmax=np.percentile(angle_flatten,99,weights=np.abs(u).flatten(),method='inverted_cdf')

    u=(u*np.exp(-1j*vmin))**(np.pi/(vmax-vmin))

    grid_xa.dat["times"]=u


    #plt.imshow(np.angle(u),alpha=np.abs(u)/(np.abs(u).max()),vmin=0,vmax=0.5*np.pi,cmap='hsv')
    #plt.colorbar()
    #plt.show()

    if (side=='right'):
        im_mask = ((grid_tof_segm.dat["0"]==dict_tof_seg['right_ica'])+(grid_tof_segm.dat["0"]==dict_tof_seg['right_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['anterior'])>0).astype('float')
    else:
        im_mask = ((grid_tof_segm.dat["0"]==dict_tof_seg['left_ica'])+(grid_tof_segm.dat["0"]==dict_tof_seg['left_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['anterior'])>0).astype('float')

    im = grid_tof.dat["0"]*im_mask

    im_mask=im_mask.astype(bool)

    phi = np.ma.MaskedArray(np.ones(im_mask.shape), mask=~im_mask)
    sum_im_mask=np.sum(im_mask.astype(int),axis=(0,1))
    lowest_points= np.min(np.nonzero(sum_im_mask))

    starting_points=np.zeros(im_mask.shape)
    starting_points[:,:,lowest_points]=im_mask.astype(int)[:,:,lowest_points]

    phi[starting_points>0] = 0

    h=np.linalg.norm(grid_tof.affine[:3,:3],axis=0)

    arrival_time = skfmm.distance(phi,dx=h)
    Tmax=arrival_time.max()

    grid_tof.dat["times"]=im*np.exp(1j*np.pi*arrival_time/Tmax).filled(0)

    return grid_xa, grid_tof





"""
    for d in grid_xa.dat:
        grid_xa.dat["carotid"]=grid_xa.dat[xa_frame_carotid]*grid_xa_mask.dat[xa_frame_carotid]


    if (side=='right'):
        grid_tof.dat["carotid"] = (
            (grid_tof_segm.dat["0"] == 2) + (grid_tof_segm.dat["0"] == 4) > 0
        ).astype("float")
    else:
        grid_tof.dat["carotid"] = (
            (grid_tof_segm.dat["0"] == 1) + (grid_tof_segm.dat["0"] == 4) > 0
        ).astype("float")

    grid_tof.dat["carotid"] *= grid_tof.dat["0"]

    grid_tof.dat["all"] = (grid_tof_segm.dat["0"] > 0).astype("float")
    grid_tof.dat["all"] *= grid_tof.dat["0"]
"""

def load_data_raw(path_xa_front,path_xa_front_metadata, path_xa_sagit,path_xa_sagit_metadata,path_tof, path_tof_segm):

    grid_xa_front = SpatialGrid_XA.fromNifti(path_xa_front, path_xa_front_metadata)
    grid_xa_sagit = SpatialGrid_XA.fromNifti(path_xa_sagit, path_xa_sagit_metadata)
    grid_tof = SpatialGridAffine.fromNifti(path_tof)
    grid_tof_segm = SpatialGridAffine.fromNifti(path_tof_segm)

    return grid_xa_front,grid_xa_sagit, grid_tof, grid_tof_segm


def pre_proc_dsa(grid_xa):
    X = np.zeros((len(grid_xa.dat), np.prod(grid_xa.shape)))
    grid_xa_new = grid_xa.copy()

    for i in range(X.shape[0]):
        grid_xa_new.dat[i] = np.clip(grid_xa.dat[0] - grid_xa.dat[i], 0, None)
        grid_xa_new.dat[i] = denoise_tv_chambolle(grid_xa_new.dat[i],weight=10)
        grid_xa_new.dat[i] = np.clip(grid_xa_new.dat[i], np.percentile(grid_xa_new.dat[i],1) , np.percentile(grid_xa_new.dat[i],99))- np.percentile(grid_xa_new.dat[i],1) 

    return grid_xa_new




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
        
def append_all_new(grid_xa_front, grid_xa_front_mask, grid_xa_sagit, grid_xa_sagit_mask, grid_tof, grid_tof_segm, side, xa_frame_mca_front, xa_frame_mca_sagit):

    grid_xa_front.dat["all"]=grid_xa_front.dat[xa_frame_mca_front]*grid_xa_front_mask.dat[xa_frame_mca_front]
    grid_xa_sagit.dat["all"]=grid_xa_sagit.dat[xa_frame_mca_sagit]*grid_xa_sagit_mask.dat[xa_frame_mca_sagit]

    if (side=='right'):
        grid_tof.dat["all"] = ((grid_tof_segm.dat["0"]==dict_tof_seg['right_ica'])+(grid_tof_segm.dat["0"]==dict_tof_seg['right_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['anterior'])>0).astype('float')
    else:
        grid_tof.dat["all"] = ((grid_tof_segm.dat["0"]==dict_tof_seg['left_ica'])+(grid_tof_segm.dat["0"]==dict_tof_seg['left_mca'])+(grid_tof_segm.dat["0"]==dict_tof_seg['anterior'])>0).astype('float')


    grid_tof.dat["all"] *= grid_tof.dat["0"]

    grid_tof.dat["all_both_sides"]=(grid_tof_segm.dat["0"]>0)*grid_tof.dat["0"]

    return grid_xa_front,grid_xa_sagit, grid_tof

