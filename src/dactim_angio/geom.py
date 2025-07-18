import numpy as np
import sys
import pyvista as pv
from scipy.spatial import KDTree
from scipy.sparse.linalg import cg
from scipy.sparse import coo_array
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from sklearn.neighbors import BallTree
from pyvista import cartesian_to_spherical
import time
import matplotlib.pyplot as plt
import pydicom as dcm

def dat_to_listdat(dat):
	return np.ravel(np.array(dat))

def listdat_to_dat(listdat,shape):
	return np.reshape(listdat,shape)

def X_to_listpoints(X):
	return np.vstack([dat_to_listdat(x) for x in X]).T

def listpoints_to_X(listpoints,shape):
	return [np.reshape(listpoints[:,i],shape) for i in range(np.array(listpoints).shape[1])]

def cart2sphere(listpoints):
    x = listpoints[:, 0]
    y = listpoints[:, 1]
    z = listpoints[:, 2]
    r,lati,longi = cartesian_to_spherical(x, y, z)
    return np.array([r,np.pi/2-lati,longi]).T

def augment(listpoints):
	return np.concatenate((listpoints,np.ones((listpoints.shape[0],1))),axis=1)

def kernel_r(r,sigma):
 	return np.exp(-r**2/sigma**2)


class GaussianPoints():
	def __init__(self,points,data,sigma):
		self.points=points
		self.N=points.shape[0]
		self.tree=KDTree(points)
		self.data=data
		self.sigma=sigma

		MyMatrix=coo_array(self.tree.sparse_distance_matrix(self.tree,5*self.sigma))
		MyMatrix.data=kernel_r(MyMatrix.data,self.sigma)

		self.weights=cg(MyMatrix,data)[0]

	def eval(self,query_points,sigma_add=0):
 		Dist=self.tree.sparse_distance_matrix(KDTree(query_points),5*(self.sigma+sigma_add))
 		MyMatrix=coo_array(Dist)
 		MyMatrix.data=kernel_r(MyMatrix.data,self.sigma+sigma_add)
 		values=MyMatrix.T@(self.weights)
 		return values

	def eval_rays(self,ray_fan,sigma_add=0):
		sigma_angle=np.asin((self.sigma+sigma_add)/np.min(np.linalg.norm(self.points-ray_fan.center,axis=1)))
		MyMatrix=ray_fan.distPointsMatrix_V2(self.points,5*sigma_angle)
		MyMatrix.data=kernel_r(MyMatrix.data,self.sigma+sigma_add)*np.sqrt(np.pi)*(self.sigma+sigma_add)
		values=MyMatrix.T@(self.weights)
		return values

	def eval_rays_shape_derivative(self,grad_T,ray_fan,sigma_add=0):
		grad_t_xi=grad_T(self.points)
		sigma_angle=np.asin((self.sigma+sigma_add)/np.min(np.linalg.norm(self.points-ray_fan.center,axis=1)))
		res=ray_fan.distPointsMatrix_deriv_V2(self.points,grad_t_xi,5*sigma_angle)
		MyMatrix=res['dists']
		MyMatrix2=res['pscal']
		MyMatrix.data=-2*(MyMatrix2.data)*kernel_r(MyMatrix.data,self.sigma+sigma_add)*np.sqrt(np.pi)/(self.sigma+sigma_add)
		values=MyMatrix.T@(self.weights)
		return values

class GaussianRays():
	def __init__(self,ray_fan,data,sigma_angle):
		self.ray_fan=ray_fan
		self.data=data
		self.sigma_angle=sigma_angle
		MyMatrix=ray_fan.distRayMatrix_V2(self.ray_fan.points,self.sigma_angle)
		MyMatrix.data=kernel_r(MyMatrix.data,self.sigma_angle)
		res=cg(MyMatrix,data)#,rtol=1e-8)
		self.weights=res[0]
		
	def eval(self,query_points,sigma_angle_add=0):
		Dist=self.ray_fan.distRayMatrix_V2(query_points,5*(self.sigma_angle+sigma_angle_add))
		MyMatrix=coo_array(Dist)
		MyMatrix.data=kernel_r(MyMatrix.data,self.sigma_angle+sigma_angle_add)
		values=MyMatrix@(self.weights)
		return values

	
	def toPyvista(self,n0,n1,radius,name):
		points_spherical=cart2sphere(self.ray_fan.points-self.ray_fan.center)
		conv=180/np.pi
		grid_scalar = pv.grid_from_sph_coords(conv*np.linspace(points_spherical[:,2].min(),points_spherical[:,2].max(),n0), 90-conv*np.linspace(points_spherical[:,1].min(),points_spherical[:,1].max(),n1), radius)
		grid_scalar.points+=self.ray_fan.center
		scalar_data=self.eval(grid_scalar.points)
		grid_scalar.point_data[name] = scalar_data
		return grid_scalar


def normalize(vectors):
	N=vectors.shape[0]
	norms=np.linalg.norm(vectors,axis=1).reshape((N,1))
	return vectors/norms



class RayFan():
	def __init__(self,center,points):
		self.center=center
		self.points=points
		self.N=points.shape[0]
		points_spherical=cart2sphere(points-center)
		self.tree= BallTree(points_spherical[:,1:],metric='haversine')

	def distPointsMatrix(self,query_points,r_angle):
		query_points_spherical=cart2sphere(query_points-self.center)
		(ind, dist)=self.tree.query_radius(query_points_spherical[:,1:],r_angle, return_distance=True)
		radii=query_points_spherical[:,0]
		
		n_cols = self.N
		n_rows = query_points.shape[0]
		Mat = lil_matrix((n_rows, n_cols), dtype=float)
		rows  = np.array([r.tolist() for r in ind],dtype=object)
		dists = np.array([list(np.sin(dist[i])*radii[i]) for i in range(len(ind))],dtype=object)
		
		Mat = lil_matrix((n_rows, n_cols), dtype=float)
		Mat.rows = rows
		Mat.data = dists
		print(Mat)
		Mat=Mat.asformat('coo')
		return Mat

	def distPointsMatrix_V2(self,query_points,r_angle):
		query_points_spherical=cart2sphere(query_points-self.center)
		(ind, dist)=self.tree.query_radius(query_points_spherical[:,1:],r_angle, return_distance=True)
		radii=query_points_spherical[:,0]
		
		lengths = np.array([len(d) for d in dist])

		# Indices inverses pour retrouver la ligne d’origine
		row_indices = np.repeat(np.arange(len(dist)), lengths)

		# Aplatissement des distances et indices
		all_dist = np.concatenate(dist)
		all_ind = np.concatenate(ind)

		# Radii correspondant à chaque distance
		all_radii = radii[row_indices]

		# ---- dists vectorisé ----
		all_dists = np.sin(all_dist) * all_radii

		# ---- Reconstruction des matrices creuses ----

		n_rows = query_points.shape[0]
		n_cols = self.N

		Mat = coo_matrix((all_dists, (row_indices, all_ind)), shape=(n_rows, n_cols))

		return Mat


	def distPointsMatrix_deriv(self,query_points,variation_vector,r_angle):
		query_points_spherical=cart2sphere(query_points-self.center)
		(ind, dist)=self.tree.query_radius(query_points_spherical[:,1:],r_angle, return_distance=True)
		radii=query_points_spherical[:,0]

		rayUnitVectors  =normalize(self.points-self.center)
		queryVectors=query_points-self.center

		pscal_qv_vv=np.einsum('ij,ij->i', queryVectors, variation_vector)

		n_cols = self.N
		n_rows = query_points.shape[0]
		Mat = lil_matrix((n_rows, n_cols), dtype=float)
		rows  = np.array([r.tolist() for r in ind],dtype=object)

		tic=time.time()
		dists = np.array([list(np.sin(dist[i])*radii[i]) for i in range(len(ind))],dtype=object)
		toc=time.time()
		print('\t Time elapsed : ',toc-tic)

		tic=time.time()
		points_proj = np.array([list( -(variation_vector[i,:]@rayUnitVectors[ind[i],:].T)*(np.cos(dist[i])*radii[i]) + pscal_qv_vv[i] ) for i in range(len(ind))],dtype=object)
		
		toc=time.time()
		print('\t Time elapsed : ',toc-tic)

		Mat = lil_matrix((n_rows, n_cols), dtype=float)
		Mat.rows = rows
		Mat.data = dists
		Mat=Mat.asformat('coo')

		Mat2 = lil_matrix((n_rows, n_cols), dtype=float)
		Mat2.rows = rows
		Mat2.data = points_proj
		Mat2=Mat.asformat('coo')

		return {'dists':Mat,'pscal':Mat2}


	def distPointsMatrix_deriv_V2(self,query_points,variation_vector,r_angle):
		query_points_spherical=cart2sphere(query_points-self.center)
		(ind, dist)=self.tree.query_radius(query_points_spherical[:,1:],r_angle, return_distance=True)
		radii=query_points_spherical[:,0]

		rayUnitVectors  =normalize(self.points-self.center)
		queryVectors = query_points-self.center

		pscal_qv_vv=np.einsum('ij,ij->i', queryVectors, variation_vector)

		# Longueurs des listes pour reconstruction plus tard
		lengths = np.array([len(d) for d in dist])

		# Indices inverses pour retrouver la ligne d’origine
		row_indices = np.repeat(np.arange(len(dist)), lengths)

		# Aplatissement des distances et indices
		all_dist = np.concatenate(dist)
		all_ind = np.concatenate(ind)

		# Radii correspondant à chaque distance
		all_radii = radii[row_indices]

		# ---- dists vectorisé ----
		all_dists = np.sin(all_dist) * all_radii

		# ---- points_proj vectorisé ----
		all_variation = variation_vector[row_indices]  # (total_pts, 3)
		all_rays = rayUnitVectors[all_ind]             # (total_pts, 3)
		all_query= queryVectors[row_indices]
		all_rays_unit=normalize(all_rays)
		all_query_unit= normalize(all_query)

		all_cosradii = np.cos(all_dist) * all_radii

		dot_prods = np.einsum('ij,ij->i', all_variation, all_rays)

		
		all_proj = -  all_cosradii *dot_prods + pscal_qv_vv[row_indices]

		# ---- Reconstruction des matrices creuses ----

		n_rows = query_points.shape[0]
		n_cols = self.N

		Mat = coo_matrix((all_dists, (row_indices, all_ind)), shape=(n_rows, n_cols))
		Mat2 = coo_matrix((all_proj, (row_indices, all_ind)), shape=(n_rows, n_cols))

		#print("Distance Matrix")
		#print(Mat)

		#print("variation vector")
		#print(variation_vector)

		#print("Pscal Matrix")
		#print(Mat2)
		
		return {'dists':Mat,'pscal':Mat2}


	def distRayMatrix(self,query_points,r):

		query_points_spherical=cart2sphere(query_points-self.center)
		(ind, dist)=self.tree.query_radius(query_points_spherical[:,1:],r, return_distance=True)
		radii=query_points_spherical[:,0]
		row=[]
		col=[]
		val=[]
		k=0
		for idx in ind :
			row+=[k]*len(idx)
			col+=idx.tolist()
			val+=dist[k].tolist()
			k+=1
		n_cols = self.N
		n_rows = query_points.shape[0]
		Mat=coo_array((val, (row, col)), shape=(n_rows, n_cols))

		return Mat

	def distRayMatrix_V2(self,query_points,r):

		query_points_spherical=cart2sphere(query_points-self.center)
		(ind, dist)=self.tree.query_radius(query_points_spherical[:,1:],r, return_distance=True)

		# Longueurs des listes pour reconstruction plus tard
		lengths = np.array([len(d) for d in dist])

		# Indices inverses pour retrouver la ligne d’origine
		row_indices = np.repeat(np.arange(len(dist)), lengths)

		# Aplatissement des distances et indices
		all_dist = np.concatenate(dist)
		all_ind = np.concatenate(ind)

		# ---- Reconstruction des matrices creuses ----

		n_rows = query_points.shape[0]
		n_cols = self.N

		Mat = coo_matrix((all_dist, (row_indices, all_ind)), shape=(n_rows, n_cols))

		return Mat

	def exclude_close_rays(self,eps2):
		points_spherical=cart2sphere(self.points-self.center)
		(ind, dist)=self.tree.query_radius(points_spherical[:,1:], eps2, return_distance=True)
		kept=[]
		thrown=[]
		current=0
		for idx in ind:
			if current not in thrown:
				kept.append(current)
				thrown+=idx.tolist()
			current+=1
		self.__init__(self.center,self.points[kept,:],self.eps)
	

	def toPyvista(self):
		res=pv.MultiBlock()
		for i in range(self.N):
			res.append(pv.Line(self.center,self.points[i,:]))
		return res

	def toPointCloud(self):
		return PointCloud(self.points,self.eps)


class Camera():
	def __init__(self,focal_point,up_vector,left_vector,up_angle,down_angle,left_angle,right_angle,dist):
		self.focal_point=focal_point
		self.up_vector=up_vector/np.linalg.norm(up_vector)
		self.left_vector=left_vector/np.linalg.norm(left_vector)
		self.forward_vector=np.cross(self.left_vector,self.up_vector)
		self.up_angle=up_angle
		self.down_angle=down_angle
		self.left_angle=left_angle
		self.right_angle=right_angle
		self.dist=dist
	
	@classmethod
	def fromPointClouds(cls,focal_point,up_vector,left_vector,listpoints,p=0):
		fooCam=Camera(focal_point,up_vector,left_vector,0,0,0,0,1)
		affine_cam_inv=np.linalg.inv(fooCam.getAffineCam())
		listpoints_in_cam_coord=(affine_cam_inv@augment(listpoints).T).T[:,:3]
		points_horizontal_angles=np.arctan2(listpoints_in_cam_coord[:,0],listpoints_in_cam_coord[:,2])
		points_vertical_angles=np.arctan2(listpoints_in_cam_coord[:,1],listpoints_in_cam_coord[:,2])
		up_angle=np.percentile(points_vertical_angles,100*(1-p))
		down_angle=-np.percentile(points_vertical_angles,100*p)
		left_angle=np.percentile(points_horizontal_angles,100*(1-p))
		right_angle=-np.percentile(points_horizontal_angles,100*p)
		
		dists=np.abs(np.array(fooCam.forward_vector)@(listpoints-focal_point).T)

		return cls(focal_point,up_vector,left_vector,up_angle,down_angle,left_angle,right_angle,dists.max())



	def inSight(self,listpoints):
		affine_cam_inv=np.linalg.inv(self.getAffineCam())
		listpoints_in_cam_coord=(affine_cam_inv@augment(listpoints).T).T
		
		points_horizontal_angles=np.arctan2(listpoints_in_cam_coord[:,0],listpoints_in_cam_coord[:,2])
		points_vertical_angles=np.arctan2(listpoints_in_cam_coord[:,1],listpoints_in_cam_coord[:,2])

		return np.argwhere((points_horizontal_angles<self.left_angle)
			*(points_vertical_angles<self.up_angle)
			*(points_horizontal_angles>-self.right_angle)
			*(points_vertical_angles>-self.down_angle)).flatten()


	def getAffineCam(self):
		affine_cam=np.array([self.left_vector,self.up_vector,self.forward_vector,self.focal_point]).T
		affine_cam=np.concatenate((affine_cam,[[0,0,0,1]]),axis=0)
		return affine_cam

	def toPyvista(self):
		dist=self.dist
		res=pv.MultiBlock()

		aff=self.getAffineCam()
		center=tuple(self.focal_point)
		corners=[[dist*np.tan(self.left_angle),dist*np.tan(self.up_angle),dist,1],
		[-dist*np.tan(self.right_angle),dist*np.tan(self.up_angle),dist,1],
		[-dist*np.tan(self.right_angle),-dist*np.tan(self.down_angle),dist,1],
		[dist*np.tan(self.left_angle),-dist*np.tan(self.down_angle),dist,1]]

		corners_pushed=[(aff@c)[:3] for c in corners]

		for c in corners_pushed:
			res.append(pv.Line(center,c))

		res.append(pv.Line(corners_pushed[0],corners_pushed[1]))
		res.append(pv.Line(corners_pushed[1],corners_pushed[2]))
		res.append(pv.Line(corners_pushed[2],corners_pushed[3]))
		res.append(pv.Line(corners_pushed[3],corners_pushed[0]))
		
		return pv.MultiBlock(res)


def applyAffine(affine,points):
	return ((affine@(augment(points)).T).T)[:,:3]

def getAffine(theta_x,theta_y,theta_z,sx,sy,sz):

	S=[sx,sy,sz]

	c= lambda theta : np.cos(np.pi*theta/180)
	s= lambda theta : np.sin(np.pi*theta/180)

	dc= lambda theta : -(np.pi/180)*s(theta)
	ds= lambda theta : (np.pi/180)*c(theta)

	t1=theta_x
	t2=theta_y
	t3=theta_z

	RotMat=np.array([
		[c(t2)*c(t3)                  , -c(t2)*s(t3)                 , s(t2)       ],
		[c(t1)*s(t3)+c(t3)*s(t1)*s(t2), c(t1)*c(t3)-s(t1)*s(t2)*s(t3), -c(t2)*s(t1)],
		[s(t1)*s(t3)-c(t1)*c(t3)*s(t2), c(t3)*s(t1)+c(t1)*s(t2)*s(t3), c(t1)*c(t2) ]])

	aff=np.eye(4)
	aff[:3,:3]=RotMat		
	aff[:3,3]=S


	invRotMat=np.array([
	 		[c(t3)*c(t2) , c(t3)*s(t2)*s(t1)+c(t1)*s(t3), s(t3)*s(t1)-c(t3)*c(t1)*s(t2) ],
	 		[-c(t2)*s(t3), c(t3)*c(t1)-s(t3)*s(t2)*s(t1), c(t1)*s(t3)*s(t2)+c(t3)*s(t1) ],
	 		[s(t2)       , -c(t2)*s(t1)                 , c(t2)*c(t1)                   ]])

	aff_inv=np.eye(4)
	aff_inv[:3,:3]=invRotMat
	aff_inv[:3,3]=-invRotMat@S

	return aff,aff_inv


def getAffineDerivatives(theta_x,theta_y,theta_z,sx,sy,sz):

	S=[sx,sy,sz]

	c= lambda theta : np.cos(np.pi*theta/180)
	s= lambda theta : np.sin(np.pi*theta/180)

	dc= lambda theta : -(np.pi/180)*s(theta)
	ds= lambda theta : (np.pi/180)*c(theta)

	t1=theta_x
	t2=theta_y
	t3=theta_z
	
	RotMat=np.array([
		[c(t2)*c(t3)                  , -c(t2)*s(t3)                 , s(t2)       ],
		[c(t1)*s(t3)+c(t3)*s(t1)*s(t2), c(t1)*c(t3)-s(t1)*s(t2)*s(t3), -c(t2)*s(t1)],
		[s(t1)*s(t3)-c(t1)*c(t3)*s(t2), c(t3)*s(t1)+c(t1)*s(t2)*s(t3), c(t1)*c(t2) ]])

	D0RotMat=np.array([
		[0                  , 0                 , 0       ],
		[dc(t1)*s(t3)+c(t3)*ds(t1)*s(t2), dc(t1)*c(t3)-ds(t1)*s(t2)*s(t3), -c(t2)*ds(t1)],
		[ds(t1)*s(t3)-dc(t1)*c(t3)*s(t2), c(t3)*ds(t1)+dc(t1)*s(t2)*s(t3), dc(t1)*c(t2) ]])

	D1RotMat=np.array([
		[dc(t2)*c(t3)                  , -dc(t2)*s(t3)                 , ds(t2)       ],
		[c(t3)*s(t1)*ds(t2), -s(t1)*ds(t2)*s(t3), -dc(t2)*s(t1)],
		[-c(t1)*c(t3)*ds(t2), c(t1)*ds(t2)*s(t3), c(t1)*dc(t2) ]])

	D2RotMat=np.array([
		[c(t2)*dc(t3)                  , -c(t2)*ds(t3)                 , 0       ],
		[c(t1)*ds(t3)+dc(t3)*s(t1)*s(t2), c(t1)*dc(t3)-s(t1)*s(t2)*ds(t3), 0],
		[s(t1)*ds(t3)-c(t1)*dc(t3)*s(t2), dc(t3)*s(t1)+c(t1)*s(t2)*ds(t3), 0 ]])

	
	d0_aff=np.eye(4)
	d0_aff[:3,:3]=D0RotMat

	d1_aff=np.eye(4)
	d1_aff[:3,:3]=D1RotMat

	d2_aff=np.eye(4)
	d2_aff[:3,:3]=D2RotMat

	d3_aff=np.eye(4)
	d3_aff[:3,:3]=0*d3_aff[:3,:3]
	d3_aff[:3,3]=[1,0,0]

	d4_aff=np.eye(4)
	d4_aff[:3,:3]=0*d4_aff[:3,:3]
	d4_aff[:3,3]=[0,1,0]

	d5_aff=np.eye(4)
	d5_aff[:3,:3]=0*d5_aff[:3,:3]
	d5_aff[:3,3]=[0,0,1]

	return [d0_aff,d1_aff,d2_aff,d3_aff,d4_aff,d5_aff]
