import numpy as np
import sys
import pyvista as pv
from scipy.spatial import KDTree
from scipy.sparse.linalg import cg
from scipy.sparse import coo_array
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from scipy import sparse

from sklearn.neighbors import BallTree
from pyvista import cartesian_to_spherical
import time

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

def normalize(vectors):
    N=vectors.shape[0]
    norms=np.linalg.norm(vectors,axis=1).reshape((N,1))
    return vectors/norms

class PointCloud():
    def __init__(self,points):
        self.points=points
        self.N=points.shape[0]
        self.tree=KDTree(points)

class RayFan():
    def __init__(self,center,points):
        self.center=center
        self.points=points
        self.N=points.shape[0]
        points_spherical=cart2sphere(points-center)
        self.tree= BallTree(points_spherical[:,1:],metric='haversine')


def dist_points_points_matrix(point_cloud,point_cloud_other,r_cut):
    return point_cloud.tree.sparse_distance_matrix(point_cloud_other.tree,r_cut)

def dist_rays_rays_matrix(gaussian_rays,gaussian_rays_other,r_angle_cut):

    query_points_spherical=cart2sphere(gaussian_rays_other.points-gaussian_rays.center)

    (ind, dist)=gaussian_rays.tree.query_radius(query_points_spherical[:,1:],r_angle_cut, return_distance=True)

    # Longueurs des listes pour reconstruction plus tard
    lengths = np.array([len(d) for d in dist])

    # Indices inverses pour retrouver la ligne d’origine
    row_indices = np.repeat(np.arange(len(dist)), lengths)

    # Aplatissement des distances et indices
    all_dist = np.concatenate(dist)
    all_ind = np.concatenate(ind)

    # ---- Reconstruction des matrices creuses ----

    n_rows = gaussian_rays_other.N
    n_cols = gaussian_rays.N

    dist_mat = coo_matrix((all_dist, (row_indices, all_ind)), shape=(n_rows, n_cols))

    return dist_mat


def dist_points_rays_matrix(point_cloud,ray_fan,r_angle_cut):

    query_points_spherical=cart2sphere(point_cloud.points-ray_fan.center)
    (ind, dist)=ray_fan.tree.query_radius(query_points_spherical[:,1:],r_angle_cut, return_distance=True)
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

    n_rows = point_cloud.N
    n_cols = ray_fan.N

    dist_mat = coo_matrix((all_dists, (row_indices, all_ind)), shape=(n_rows, n_cols))
    
    return dist_mat

def d_points_dist_points_rays_matrix(point_cloud,ray_fan,r_angle_cut):

    query_points_spherical=cart2sphere(point_cloud.points-ray_fan.center)
    (ind, dist)=ray_fan.tree.query_radius(query_points_spherical[:,1:],r_angle_cut, return_distance=True)
    radii=query_points_spherical[:,0]

    rayVectors  = ray_fan.points-ray_fan.center
    queryVectors = point_cloud.points-ray_fan.center

    # Longueurs des listesoverl pour reconstruction plus tard
    lengths = np.array([len(d) for d in dist])

    # Indices inverses pour retrouver la ligne d’origine
    row_indices = np.repeat(np.arange(len(dist)), lengths)
    all_ind = np.concatenate(ind)

    # Aplatissement des distances et indices
    all_thetas = np.concatenate(dist)
    all_radii  = radii[row_indices]

    all_ray_vectors = rayVectors[all_ind]                            # (total_pts, 3)

    Ntot=all_ray_vectors.shape[0]

    all_thetas=all_thetas.reshape((Ntot,1))
    all_radii=all_radii.reshape((Ntot,1))

    all_ray_vectors_norm=np.linalg.norm(all_ray_vectors,axis=1).reshape((Ntot,1))

    all_query_vectors= queryVectors[row_indices]

    all_query_vectors_norm=np.linalg.norm(all_query_vectors,axis=1).reshape((Ntot,1))
    

    all_norm_cos=(all_query_vectors_norm/all_ray_vectors_norm)*np.cos(all_thetas)

    all_dists = np.sin(all_thetas) * all_radii

    all_var=all_query_vectors-all_norm_cos*all_ray_vectors

    all_var_x=all_var[:,0]
    all_var_y=all_var[:,1]
    all_var_z=all_var[:,2]

    all_dists=all_dists.reshape(Ntot)

    all_var_x=all_var_x.reshape(Ntot)
    all_var_y=all_var_y.reshape(Ntot)
    all_var_z=all_var_z.reshape(Ntot)

    # ---- Reconstruction des matrices creuses ----

    n_rows = point_cloud.N
    n_cols = ray_fan.N

    d_point_x_deltas = coo_matrix((all_var_x, (row_indices, all_ind)), shape=(n_rows, n_cols))
    d_point_y_deltas = coo_matrix((all_var_y, (row_indices, all_ind)), shape=(n_rows, n_cols))
    d_point_z_deltas = coo_matrix((all_var_z, (row_indices, all_ind)), shape=(n_rows, n_cols))
    
    return d_point_x_deltas, d_point_y_deltas, d_point_z_deltas

def d_rays_dist_points_rays_matrix(point_cloud,ray_fan,r_angle_cut):

    query_points_spherical=cart2sphere(point_cloud-ray_fan.center)
    (ind, dist)=ray_fan.tree.query_radius(query_points_spherical[:,1:],r_angle_cut, return_distance=True)
    radii=query_points_spherical[:,0]

    rayVectors  = ray_fan.points-ray_fan.center
    queryVectors = point_cloud.points-ray_fan.center

    # Longueurs des listesoverl pour reconstruction plus tard
    lengths = np.array([len(d) for d in dist])

    # Indices inverses pour retrouver la ligne d’origine
    row_indices = np.repeat(np.arange(len(dist)), lengths)
    all_ind = np.concatenate(ind)

    # Aplatissement des distances et indices
    all_thetas = np.concatenate(dist)
    all_radii  = radii[row_indices]

    all_ray_vectors = rayVectors[all_ind]                            # (total_pts, 3)

    Ntot=all_ray_vectors.shape[0]

    all_thetas=all_thetas.reshape((Ntot,1))
    all_radii=all_radii.reshape((Ntot,1))

    all_ray_vectors_norm=np.linalg.norm(all_ray_vectors,axis=1).reshape((Ntot,1))

    all_query_vectors= queryVectors[row_indices]

    all_query_vectors_norm=np.linalg.norm(all_query_vectors,axis=1).reshape((Ntot,1))

    all_norm_cos=(all_query_vectors_norm/all_ray_vectors_norm)*np.cos(all_thetas)

    all_dists = np.sin(all_thetas) * all_radii

    all_var=-all_norm_cos*all_query_vectors+(all_norm_cos**2)*all_ray_vectors

    all_var_x=all_var[:,0]
    all_var_y=all_var[:,1]
    all_var_z=all_var[:,2]

    all_dists=all_dists.reshape(Ntot)

    all_var_x=all_var_x.reshape(Ntot)
    all_var_y=all_var_y.reshape(Ntot)
    all_var_z=all_var_z.reshape(Ntot)

    # ---- Reconstruction des matrices creuses ----

    n_rows = point_cloud.N
    n_cols = ray_fan.N

    d_point_x_deltas = coo_matrix((all_var_x, (row_indices, all_ind)), shape=(n_rows, n_cols))
    d_point_y_deltas = coo_matrix((all_var_y, (row_indices, all_ind)), shape=(n_rows, n_cols))
    d_point_z_deltas = coo_matrix((all_var_z, (row_indices, all_ind)), shape=(n_rows, n_cols))
    
    return d_point_x_deltas, d_point_y_deltas, d_point_z_deltas


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
        [0                              , 0                              , 0             ],
        [dc(t1)*s(t3)+c(t3)*ds(t1)*s(t2), dc(t1)*c(t3)-ds(t1)*s(t2)*s(t3), -c(t2)*ds(t1) ],
        [ds(t1)*s(t3)-dc(t1)*c(t3)*s(t2), c(t3)*ds(t1)+dc(t1)*s(t2)*s(t3), dc(t1)*c(t2)  ]])

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
