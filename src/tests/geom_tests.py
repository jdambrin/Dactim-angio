import sys
sys.path.append('../')
print(sys.path)
from dactim_angio.geom import GaussianPointData
from dactim_angio.geom import PointCloud
from dactim_angio.geom import RayFan
from dactim_angio.geom import Camera
import numpy as np
import pyvista as pv

def testPointCloudPV():

	points=np.random.rand(100,3)

	cloud=PointCloud(points,0.1)

	pv_object=cloud.toPyvista()

	p = pv.Plotter(window_size=[2000, 2000])

	p.add_mesh(pv_object,color="black", point_size=10, render_points_as_spheres=True)

	p.show_grid()
	p.show()


def testRayFanPV():

	points=np.random.rand(100,3)

	fan=RayFan([0,0,0],points,0.1)

	pv_objects=fan.toPyvista()

	p = pv.Plotter(window_size=[2000, 2000])

	for o in pv_objects:
		p.add_mesh(o,color="black")

	p.show_grid()
	p.show()



def testCameraPV():

	points=np.random.rand(10000,3)
	cloud=PointCloud(points,0.1)
	pv_cloud=cloud.toPyvista()


	v=np.array([[1,1,1],[1,0,0],[0,1,0]]).T

	(Q,R)=np.linalg.qr(v)

	c=Camera([0,0,0],Q[:,1],Q[:,2],0.05*np.pi,0.1*np.pi,0.2*np.pi,0.1*np.pi)
	pv_objects=c.toPyvista(1.5)

	print(c.inSight(points))
	points_selected=points[c.inSight(points),:]


	cloud2=PointCloud(points_selected,0.1)
	pv_cloud2=cloud2.toPyvista()

	p = pv.Plotter(window_size=[2000, 2000])

	p.add_mesh(o,color="black",line_width=10)

	p.add_mesh(pv_cloud,color="black", point_size=10, render_points_as_spheres=True)
	p.add_mesh(pv_cloud2,color="red", point_size=10, render_points_as_spheres=True)

	p.show_grid()
	p.show()


def testDefCameraFromPoints():

	points=np.random.rand(1000,3)
	cloud=PointCloud(points,0.1)
	pv_cloud=cloud.toPyvista()

	v=np.array([[1,1,1],[1,0,0],[0,1,0]]).T
	(Q,R)=np.linalg.qr(v)

	c=Camera.fromPointClouds([0,0,0],Q[:,1],Q[:,2],points,0.2)

	pv_camera=c.toPyvista(1.5)

	points_selected=points[c.inSight(points),:]
	cloud2=PointCloud(points_selected,0.1)
	pv_cloud2=cloud2.toPyvista()

	p = pv.Plotter(window_size=[2000, 2000])

	p.add_mesh(pv_camera,color="black",line_width=10)

	p.add_mesh(pv_cloud,color="black", point_size=10, render_points_as_spheres=True)
	p.add_mesh(pv_cloud2,color="red", point_size=10, render_points_as_spheres=True)

	p.show_grid()
	p.show()



def testGaussianPointData():

	cloud=PointCloud(np.random.rand(100,3),0.1)
	rays=RayFan([0,0,0],np.random.rand(10,3),0.1)
	data=GaussianPointData(cloud,50*np.random.rand(100),0.01)
	res=data.eval_rays(rays)










if __name__=='__main__':
	testDefCameraFromPoints()
