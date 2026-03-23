import numpy as np
import pyvista as pv
from scipy.sparse.linalg import cg
from scipy.sparse import coo_array
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from scipy import sparse

from dactim_angio.geom import PointCloud,RayFan
from dactim_angio.geom import dist_points_points_matrix
from dactim_angio.geom import dist_rays_rays_matrix
from dactim_angio.geom import dist_points_rays_matrix
from dactim_angio.geom import d_points_dist_points_rays_matrix

from dactim_angio.geom import cart2sphere


def kernel_r(r,sigma):
    return np.exp(-r**2/sigma**2)


class GaussianRBF():
    def __init__(self,points,sigma,data={}):

        self.point_cloud=PointCloud(points)
        self.data=data
        self.sigma=sigma

        self.weights={}

        if(data):
            K_matrix=coo_array(self.point_cloud.tree.sparse_distance_matrix(self.point_cloud.tree,5*self.sigma))
            K_matrix.data=kernel_r(K_matrix.data,self.sigma)
            for key in data:
                self.weights[key]=cg(K_matrix,data[key])[0]

    def eval(self,query_point_cloud,field=0):
        dist=dist_points_points_matrix(self.point_cloud,query_point_cloud,5*self.sigma)
        K_matrix=coo_array(dist)
        K_matrix.data=kernel_r(K_matrix.data,self.sigma)
        values=K_matrix.T@(self.weights[field])
        return values

    def eval_rays(self,query_ray_fan,field=0):
        sigma_angle=np.asin((self.sigma)/np.min(np.linalg.norm(self.point_cloud.points-query_ray_fan.center,axis=1)))
        K_delta_matrix=dist_points_rays_matrix(self.point_cloud,query_ray_fan,5*sigma_angle)
        K_delta_matrix.data=kernel_r(K_delta_matrix.data,self.sigma)*np.sqrt(np.pi)*(self.sigma)
        values=K_delta_matrix.T@(self.weights[field])
        return values

    def d_points_eval_rays(self,query_ray_fan,field=0):

        sigma_angle=np.asin((self.sigma)/np.min(np.linalg.norm(self.point_cloud.points-query_ray_fan.center,axis=1)))

        res=d_points_dist_points_rays_matrix(self.point_cloud,query_ray_fan,5*sigma_angle)
        deltas=dist_points_rays_matrix(self.point_cloud,query_ray_fan,5*sigma_angle)

        dx_deltas=res[0]
        dy_deltas=res[1]
        dz_deltas=res[2]

        d_point_eval_rays=[]

        MyMatrix_x=deltas.copy()
        MyMatrix_x.data=-2*(dx_deltas.data)*kernel_r(deltas.data,self.sigma)*np.sqrt(np.pi)/(self.sigma)
        MyMatrix_x=sparse.diags(self.weights[field])@MyMatrix_x

        d_point_eval_rays.append(MyMatrix_x.T)

        MyMatrix_y=deltas.copy()
        MyMatrix_y.data=-2*(dy_deltas.data)*kernel_r(deltas.data,self.sigma)*np.sqrt(np.pi)/(self.sigma)
        MyMatrix_y=sparse.diags(self.weights[field])@MyMatrix_y

        d_point_eval_rays.append(MyMatrix_y.T)

        MyMatrix_z=deltas.copy()
        MyMatrix_z.data=-2*(dz_deltas.data)*kernel_r(deltas.data,self.sigma)*np.sqrt(np.pi)/(self.sigma)
        MyMatrix_z=sparse.diags(self.weights[field])@MyMatrix_z

        d_point_eval_rays.append(MyMatrix_z.T)

        return d_point_eval_rays


class GaussianRBF_rays():
    def __init__(self,ray_fan,sigma_angle,data={}):
        self.ray_fan=ray_fan
        self.data=data
        self.sigma_angle=sigma_angle

        self.weights={}

        if(data):
            K_matrix=coo_array(dist_rays_rays_matrix(self.ray_fan,self.ray_fan,5*self.sigma_angle))
            K_matrix.data=kernel_r(K_matrix.data,self.sigma_angle)
            for key in data:
                self.weights[key]=cg(K_matrix,data[key])[0]

        
    def eval(self,query_ray_fan,field):
        dist=dist_rays_rays_matrix(self.ray_fan,query_ray_fan)
        K_matrix=coo_array(dist)
        K_matrix.data=kernel_r(K_matrix.data,self.sigma_angle)
        values=K_matrix@(self.weights[field])
        return values

    def eval_shape_derivative(self,GRAD_T_points,GRAD_T_rays,ray_fan_data,sigma_angle_add=0):
        pass
    
    def toPyvista(self,n0,n1,radius,name):
        points_spherical=cart2sphere(self.ray_fan.points-self.ray_fan.center)
        conv=180/np.pi
        grid_scalar = pv.grid_from_sph_coords(conv*np.linspace(points_spherical[:,2].min(),points_spherical[:,2].max(),n0), 90-conv*np.linspace(points_spherical[:,1].min(),points_spherical[:,1].max(),n1), radius)
        grid_scalar.points+=self.ray_fan.center
        scalar_data=self.eval(grid_scalar.points)
        grid_scalar.point_data[name] = scalar_data
        return grid_scalar

