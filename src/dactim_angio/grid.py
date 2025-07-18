import numpy as np
import pyvista as pv
from scipy.interpolate import interpn
from pyvista.plotting.themes import DocumentTheme
import copy
import pydicom as dcm
import sys
from scipy.spatial.transform import Rotation as R
from abc import ABC, abstractmethod
from scipy.ndimage import gaussian_filter
import sys
sys.path.append('../')
import nibabel




class SpatialGrid(ABC):
	def __init__(self,shape,dat={}):
		self.shape=shape
		self.dat=dat.copy()
		self.input_dim=len(list(shape))

	@abstractmethod
	def push(self,I):
		pass

	@abstractmethod
	def pull(self,X):
		pass

	def grad_push(self,I):
		delta=1e-6
		Grad=np.zeros((3,)+(self.input_dim,)+self.shape)
		for i in range(self.input_dim):
			Ip=I.copy()
			Ip[i]+=delta
			Im=I.copy()
			Im[i]-=delta
			Grad[:,i]=(self.push(Ip)-self.push(Im))/(2*delta)
		return Grad

	def grad_pull(self,X):
		delta=1e-6 # à scaler sur le pas
		Grad=np.zeros((3,)+(self.input_dim,)+self.shape)
		for i in range(self.input_dim):
			Xp=X.copy()
			Xp[i]+=delta
			Xm=X.copy()
			Xm[i]-=delta
			Grad[:,i]=(self.pull(Xp)-self.pull(Xm))/(2*delta)
		return Grad

	def __repr__(self):
		
		s='Type: '+self.__class__.__name__+'\n'
		s+='Shape:'+str(self.shape)+'\nScalar fields:\n'
		for k in self.dat:
			s+='\t'+str(k)+': shape '+str(self.dat[k].shape)+' min/max '+str(np.min(self.dat[k]))+'/'+str(np.max(self.dat[k]))+'\n'
		return s

	def copy(self):
		
		return copy.deepcopy(self)

	def rescale(self,new_shape):

		old=self.copy()

		scale_factor=[(old.shape[i]-1)/(new_shape[i]-1) for i in range(len(old.shape))]

		Mat=np.diag(scale_factor)

		new_push=lambda I : old.push(np.tensordot(Mat,I,axes=(1,0)))
		new_pull=lambda X : np.tensordot(np.linalg.inv(Mat),old.pull(X),axes=(1,0))

		self.shape=new_shape
		self.push=new_push
		self.pull=new_pull
		self.data={}.copy()

		for f in old.dat:
			self.transferScalarData(old,fieldName=f,fill_value=0)

	def getMonotonicIndices(self):
		return [np.linspace(0,n-1,n) for n in self.shape]

	def getIndices(self):
		listargs=self.getMonotonicIndices()
		I=np.meshgrid(*listargs,indexing='ij')
		I=np.array(I)
		return I

	def getCoordinates(self):
		I=self.getIndices()
		return self.push(I)

	def getExtent(self):
		X=self.getCoordinates()
		return [(np.min(X[i]),np.max(X[i])) for i in range(3)]

	def getDim(self):
		return len(self.shape)

	def toPyvista(self):
		X=self.getCoordinates()
		grid = pv.StructuredGrid(X[0],X[1],X[2])

		for d in self.dat:
			grid.point_data[str(d)]=self.dat[d].flatten(order='F')

		return grid

	def transferAllScalarData(self,grid2,fill_value=0,sigma=None,method='linear'):
		for k in grid2.dat.keys():
			self.transferScalarData(grid2,fieldName=k,fill_value=fill_value,sigma=sigma,method=method)

	def transferScalarData(self,grid2,fieldName=None,fill_value=0,sigma=None,method='linear'):

		if(fieldName==None):
			fieldName=list(grid2.dat)[0]
		
		dat=grid2.dat[fieldName].copy()

		if(not(sigma==None)): 
			dat=gaussian_filter(dat, sigma)

		Ind_dat=grid2.getMonotonicIndices()

		Npoints=np.prod(self.shape)

		Ind_query=grid2.pull(self.getCoordinates()).reshape(self.input_dim,Npoints).T
		new_dat=interpn(tuple(Ind_dat),dat,Ind_query,bounds_error=False,fill_value=fill_value,method=method)
		self.dat[fieldName]=new_dat.reshape(self.shape)


class SpatialGridAffine(SpatialGrid):

	def __init__(self,shape,affine,dat={}):
		assert(affine.shape[0]-1==3)
		self.affine=affine
		super().__init__(shape,dat)

	@classmethod
	def fromNifti(cls,filename):
		foo=nibabel.load(filename)
		return cls(foo.get_fdata().shape,foo.affine,{'0':foo.get_fdata()})

	def getH(self):
		return np.linalg.norm(self.affine,axis=0)[:-1]

	def push(self,I):
		I=np.append(I,np.ones((1,)+I.shape[1:]),axis=0)
		I=np.tensordot(self.affine,I,axes=(1,0))
		return np.array([I[i] for i in range(3)])

	def pull(self,X):

		affine_inject=np.eye(4)
		affine_inject[:,-1]=self.affine[:,-1]

		normal=np.cross(self.affine[:3,0],self.affine[:3,1])
		affine_inject[:3,2]=normal

		affine_inject[:,:self.affine.shape[1]-1]=self.affine[:,:self.affine.shape[1]-1]

		inv_affine_inject=np.linalg.inv(affine_inject)

		X=np.append(X,np.ones((1,)+X.shape[1:]),axis=0)
		X=np.tensordot(inv_affine_inject,X,axes=(1,0))
		return np.array([X[i] for i in range(len(self.shape))])



class SpatialGridPerspective(SpatialGridAffine):

	def __init__(self,shape,affine,center,dat={}):
		assert(affine.shape[1]==3)
		self.center=center
		super().__init__(shape,affine,dat)

	@classmethod
	def fromNifti(cls,filename):
		foo=nibabel.load(filename)
		aff=foo.affine
		
		shape=foo.get_fdata().shape[:2]

		v0=aff[:3,0]/np.linalg.norm(aff[:3,0])
		v1=aff[:3,1]/np.linalg.norm(aff[:3,1])
		v2=-np.cross(v0,v1)

		new_aff=np.zeros((4,3))
		new_aff[:,:2]=aff[:,:2]
		new_aff[:,2]=aff[:,3]


		center=v2*float(foo.header['descrip'])

		data={}

		for i in range(foo.get_fdata().shape[-1]):
			data[i]=foo.get_fdata()[:,:,0,i]

		return cls(shape,new_aff,center,data)



	@classmethod
	def fromDicom(cls,filename):
		dat=dcm.dcmread(filename)
		ci=0.5*dat.pixel_array.shape[1]
		cj=0.5*dat.pixel_array.shape[2]

		affine=np.eye(3)
		affine[0,2]=-ci
		affine[1,2]=-cj
		affine[2,2]=0

		hi=float(dat.ImagerPixelSpacing[0])
		hj=float(dat.ImagerPixelSpacing[1])

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

		r = R.from_euler('xz', [dat.PositionerSecondaryAngle,dat.PositionerPrimaryAngle], degrees=True)

		affine=r.as_matrix()@affine

		aff=np.eye(4,3)
		aff[:3,:]=affine
		aff[3,2]=1

		v0=aff[:3,0]/np.linalg.norm(aff[:3,0])
		v1=aff[:3,1]/np.linalg.norm(aff[:3,1])
		v2=-np.cross(v0,v1)

		center=v2*dat.DistanceSourceToPatient

		image=dat.pixel_array
		image=image.astype('float')

		image_dict={}
		for i in range(image.shape[0]):
			image_dict[i]=image[i,:,:]

		foo=cls(image[0,:,:].shape,aff,center)

		foo.dat=image_dict
		return foo


	
# 	def __init__(self,shape,focal=(0,0,0),forward=(1,0,0),up=(0,0,1),left=(0,1,0),angle_up=10,angle_down=10,angle_left=10,angle_right=10,start=0.5,end=1,dat={}):

# 		self.focal=focal
# 		self.forward=forward
# 		self.up=up
# 		self.left=left
# 		self.angle_up=angle_up
# 		self.angle_down=angle_down
# 		self.angle_left=angle_left
# 		self.angle_right=angle_right
# 		self.start=start
# 		self.end=end
# 		super().__init__(shape,dat)

# 	def push(self,I):
# 		X=I.copy()
# 		ratio_up=np.tan(self.angle_up*(np.pi/180))
# 		ratio_down=np.tan(self.angle_down*(np.pi/180))
# 		ratio_left=np.tan(self.angle_left*(np.pi/180))
# 		ratio_right=np.tan(self.angle_right*(np.pi/180))

# 		M0=np.array([list(self.forward),list(self.left),list(self.up)]).T
# 		Q,R=np.linalg.qr(M0)
# 		Q[:,0]=Q[:,0]*np.sign(R[0,0])
# 		Q[:,1]=Q[:,1]*np.sign(R[1,1])
# 		Q[:,2]=Q[:,2]*np.sign(R[2,2])
			
# 		X[0]=X[0]/(self.shape[0]-1)
# 		X[1]=X[1]/(self.shape[1]-1)
# 		X[2]=X[2]/(self.shape[2]-1)
	
# 		X[1]=X[1]*((1-X[0])*(ratio_left+ratio_right)*self.start+X[0]*(ratio_left+ratio_right)*self.end)-((1-X[0])*ratio_left*self.start+X[0]*ratio_left*self.end)
# 		X[2]=X[2]*((1-X[0])*(ratio_up+ratio_down)*self.start+X[0]*(ratio_up+ratio_down)*self.end)-((1-X[0])*ratio_up*self.start+X[0]*ratio_up*self.end)
# 		X[0]=X[0]*(self.end-self.start)+self.start

# 		X=np.tensordot(Q,X,axes=(1,0))

# 		X[0]+=self.focal[0]
# 		X[1]+=self.focal[1]
# 		X[2]+=self.focal[2]
			
# 		return X


# 	def pull(self,X):
				
# 		I=X.copy()

# 		ratio_up=np.tan(self.angle_up*(np.pi/180))
# 		ratio_down=np.tan(self.angle_down*(np.pi/180))
# 		ratio_left=np.tan(self.angle_left*(np.pi/180))
# 		ratio_right=np.tan(self.angle_right*(np.pi/180))

# 		M0=np.array([list(self.forward),list(self.left),list(self.up)]).T
# 		Q,R=np.linalg.qr(M0)
# 		Q[:,0]=Q[:,0]*np.sign(R[0,0])
# 		Q[:,1]=Q[:,1]*np.sign(R[1,1])
# 		Q[:,2]=Q[:,2]*np.sign(R[2,2])
			
# 		I[0]-=self.focal[0]
# 		I[1]-=self.focal[1]
# 		I[2]-=self.focal[2]

# 		I=np.tensordot(Q.T,I,axes=(1,0))

# 		I[0]=(I[0]-self.start)/(self.end-self.start)
# 		I[1]=(I[1]+((1-I[0])*ratio_left*self.start+I[0]*ratio_left*self.end))/((1-I[0])*(ratio_left+ratio_right)*self.start+I[0]*(ratio_left+ratio_right)*self.end)
# 		I[2]=(I[2]+((1-I[0])*ratio_up*self.start+I[0]*ratio_up*self.end))/((1-I[0])*(ratio_up+ratio_down)*self.start+I[0]*(ratio_up+ratio_down)*self.end)

# 		I[0]=I[0]*(self.shape[0]-1)
# 		I[1]=I[1]*(self.shape[1]-1)
# 		I[2]=I[2]*(self.shape[2]-1)

# 		return I

# 	def squish(self,dist,fun='default'):

# 		shape_new=(self.shape[1],self.shape[2])

# 		focal=self.focal
# 		forward=self.forward
# 		up=self.up
# 		left=self.left
# 		angle_up=self.angle_up
# 		angle_down=self.angle_down
# 		angle_left=self.angle_left
# 		angle_right=self.angle_right
# 		start=self.start
# 		end=self.end

# 		ratio_up=np.tan(self.angle_up*(np.pi/180))
# 		ratio_down=np.tan(self.angle_down*(np.pi/180))
# 		ratio_left=np.tan(self.angle_left*(np.pi/180))
# 		ratio_right=np.tan(self.angle_right*(np.pi/180))

# 		M0=np.array([list(self.forward),list(self.left),list(self.up)]).T
# 		Q,R=np.linalg.qr(M0)
# 		Q[:,0]=Q[:,0]*np.sign(R[0,0])
# 		Q[:,1]=Q[:,1]*np.sign(R[1,1])
# 		Q[:,2]=Q[:,2]*np.sign(R[2,2])
			

# 		M1=np.array([[0,0,0],[1./(shape_new[0]-1),0,0],[0,1./(shape_new[1]-1),0],[0,0,1]])
# 		M2=np.array([[1,0,0,dist],[0,(ratio_left+ratio_right)*dist,0,-ratio_left*dist],[0,0,(ratio_up+ratio_down)*dist,-ratio_up*dist],[0,0,0,1]])
# 		M3=np.eye(4)
# 		M3[:3,:3]=Q
# 		M4=np.array([[1,0,0,focal[0]],[0,1,0,focal[1]],[0,0,1,focal[2]],[0,0,0,1]])

# 		#print(M1,M2,M3,M4)

# 		affine_new=M4@M3@M2@M1

# 		dat_new={}.copy()

# 		X=self.getCoordinates()

# 		steps=np.sqrt((X[0][0,:,:]-X[0][1,:,:])**2+(X[1][0,:,:]-X[1][1,:,:])**2+(X[2][0,:,:]-X[2][1,:,:])**2)

# 		if(fun=='default'):
# 			for f in self.dat:
# 				dat_new[f]=np.sum(self.dat[f],axis=0)*steps
# 		elif(fun=='mip'):
# 			for f in self.dat:
# 				dat_new[f]=np.max(self.dat[f],axis=0)
# 		else:
# 			for f in self.dat:
# 				dat_new[f]=fun(self.dat[f])*steps

# 		return SpatialGridAffine(shape_new,affine_new,dat_new)


class SpatialGridPushPull(SpatialGrid):
	def __init__(self,shape,push_pull,dat={}):
		self.push_pull = push_pull
		super().__init__(shape,dat)

	def push(self,I):
		return self.push_pull[0](I)

	def pull(self,X):
		return self.push_pull[1](X)



def getExtent(grid,fieldname,direc,alpha,Nsamples=100):

	direc=np.array(direc)/np.linalg.norm(direc)
	X=grid.getCoordinates()

	L=np.tensordot(direc,X,axes=(0,0))
	Lvalues=np.linspace(np.min(L),np.max(L),Nsamples)
	
	cumsum=np.zeros(Nsamples)
	for i in range(1,Nsamples):
		cumsum[i]=cumsum[i-1]+np.sum(grid.dat[fieldname][((L<Lvalues[i])*(L>=Lvalues[i-1])).astype('bool')])

	cumsum=cumsum/np.max(cumsum)

	indm=np.min(np.argwhere(cumsum>alpha))
	indp=np.max(np.argwhere(cumsum<1-alpha))

	return(Lvalues[indm],Lvalues[indp])





