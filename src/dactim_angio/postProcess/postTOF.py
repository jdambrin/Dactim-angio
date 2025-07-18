from scipy.spatial.transform import Rotation as R
import numpy as np
import pydicom as dcm
from sklearn.mixture import GaussianMixture
from registration import myreg
import matplotlib.pyplot as plt
import sys
from matplotlib import cm
import matplotlib.backends.backend_pdf
from matplotlib.colors import ListedColormap
import math

def maskBG(seq):

    (N3,N1,N2)=seq.shape
    gm = GaussianMixture(n_components=5, random_state=0).fit(seq[0,:,:].flatten().reshape(-1, 1))

    prob=gm.predict_proba(seq[:,:,:].flatten().reshape(-1, 1))

    im=np.argmin(gm.means_)
    ip=np.argmax(gm.means_)

    Mask=(1-prob[:,im].reshape(N3,N1,N2))*(1-prob[:,ip].reshape(N3,N1,N2))
    maxx=np.max(seq)

    seq=seq*Mask+(1-Mask)*maxx
    return seq

def autoCoReg(seq):
    (N3,N1,N2)=seq.shape
    seq2=seq
    seq3=1*seq2[:,:,:]
    for k in range(N3):
        print("registration image : ", k)
        seq3[k,:,:]=myreg(seq2[0,:,:],seq3[k,:,:],np.max(seq))
    return seq3

def XA_to_DSA(seq):
    (N3,N1,N2)=seq.shape
    seq4=1*seq
    for k in range(N3):
        seq4[k,:,:]=seq[0,:,:]-seq4[k,:,:]
    seq4=np.clip(seq4,0,None)    
    return seq4

def reportStack(B,pdf,minn=np.nan,maxx=np.nan,cmap=cm.gray):
    if(np.isnan(minn)):
        minn=np.min(B)
    if(np.isnan(maxx)):
        maxx=np.max(B)

    (N3,N1,N2)=B.shape
    Nfigures=math.ceil(N3/8)
    print(Nfigures, 'figures')

    for k in range(Nfigures):
        f,ax=plt.subplots(2,4,figsize=(12,6))
        for i in range(2):
            for j in range(4):
                index=j+4*i+8*k
                ax[i,j].get_xaxis().set_visible(False)
                ax[i,j].get_yaxis().set_visible(False)
                ax[i,j].axis('equal')
                if (index<N3):
                    h=ax[i,j].imshow(B[index,:,:],cmap=cmap)
                    h.set_clim(minn,maxx)
                    ax[i,j].set_title(str(index+1)+'/'+str(N3))
                else:
                    ax[i,j].set_title(' ')
                    ax[i,j].axis('off')
        f.tight_layout()
        pdf.savefig(f,dpi=150)
    return pdf


def buildAffine_XA(dat,primary_mod=0):

    ci=0.5*dat.pixel_array.shape[1]
    cj=0.5*dat.pixel_array.shape[2]

    affine=np.eye(3)
    affine[0,2]=-ci
    affine[1,2]=-cj
    affine[2,2]=0

    hi=float(dat.ImagerPixelSpacing[0])#*float(dat.DistanceSourceToPatient)/float(dat.DistanceSourceToDetector)
    hj=float(dat.ImagerPixelSpacing[1])#*float(dat.DistanceSourceToPatient)/float(dat.DistanceSourceToDetector)

    scale=np.eye(3)
    scale[0,0]=-hi
    scale[1,1]=-hj

    affine=scale@affine

    affine[2,2]=float(dat.DistanceSourceToDetector)-float(dat.DistanceSourceToPatient)

    frontal=np.zeros((3,3))
    frontal[2,0]=1
    frontal[0,1]=1
    frontal[1,2]=1

    affine=frontal@affine

    r = R.from_euler('xz', [dat.PositionerSecondaryAngle,dat.PositionerPrimaryAngle-primary_mod], degrees=True)

    affine=r.as_matrix()@affine

    aff=np.eye(4,3)
    aff[:3,:]=affine
    aff[3,2]=1

    return aff

def buildAffine_XA_new(dat,primary_mod=0):

    ci=0.5*dat.pixel_array.shape[1]
    cj=0.5*dat.pixel_array.shape[2]

    affine=np.eye(3)
    affine[0,2]=-ci
    affine[1,2]=-cj
    affine[2,2]=0

    hi=float(dat.ImagerPixelSpacing[0])#*float(dat.DistanceSourceToPatient)/float(dat.DistanceSourceToDetector)
    hj=float(dat.ImagerPixelSpacing[1])#*float(dat.DistanceSourceToPatient)/float(dat.DistanceSourceToDetector)

    scale=np.eye(3)
    scale[0,0]=-hi
    scale[1,1]=-hj

    affine=scale@affine

    affine[2,2]=float(dat.DistanceSourceToDetector)-float(dat.DistanceSourceToPatient)

    frontal=np.zeros((3,3))
    frontal[2,0]=1
    frontal[0,1]=1
    frontal[1,2]=1

    affine=frontal@affine

    r = R.from_euler('xz', [dat.PositionerSecondaryAngle,dat.PositionerPrimaryAngle-primary_mod], degrees=True)

    affine=r.as_matrix()@affine

    aff=np.eye(4,3)
    aff[:3,:]=affine
    aff[3,2]=1

    return aff



