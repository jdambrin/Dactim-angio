import sys
sys.path.append('../')
print(sys.path)
from dactim_angio.grid import SpatialGrid
from dactim_angio.grid import SpatialGridAffine
from dactim_angio.grid import SpatialGridPerspective
from dactim_angio.grid import SpatialGridPushPull
from dactim_angio.reg2D3D import RBFsparseDat
from dactim_angio.reg2D3D import dat_to_listdat
from dactim_angio.reg2D3D import listdat_to_dat
from dactim_angio.reg2D3D import X_to_listpoints
from dactim_angio.reg2D3D import listpoints_to_X
from dactim_angio.reg2D3D import Bundle

import numpy as np
import pyvista as pv

import time

# Twists coordinates around the 3rd axis
def twist(X,rate):
	Z=X.copy()
	theta=np.arctan2(X[0],X[1])
	rho=np.sqrt(X[0]**2+X[1]**2)
	theta_new=theta+rate*Z[2]
	Z[0]=rho*np.cos(theta_new)
	Z[1]=rho*np.sin(theta_new)
	return Z

# Shift coordinates with (a,b,c)
def shift(X,a):
	Y=X.copy()
	Y[0]+=a[0]
	Y[1]+=a[1]
	Y[2]+=a[2]
	return Y

def testSimpleGrid():

	# Create a simple grid, with Identity push and pull functions
	straight=SpatialGridAffine((100,100,100),np.eye(4),dat={'dull':np.ones((100,100,100))})

	# Add some scalar data (a ball centered on (20,50,30) with radius 10)
	X=straight.getCoordinates()
	straight.dat['mydata']=(np.sqrt((X[0]-20)**2+(X[1]-50)**2+(X[2]-30)**2)<10).astype('float')

	# Print the object representation
	print(straight)

	# Plot everything
	p = pv.Plotter(window_size=[2000, 2000])

	grid=straight.toPyvista()
	p.add_mesh(grid,style='wireframe', color="red",opacity=0.1)
	p.add_mesh(grid.contour([0.5],grid.point_data['mydata']), color="red",copy_mesh=True)
	p.show_grid()
	p.show()

def testTwistedGrid():

	
	# Create a grid for which the push/pull are given (with the twist function above)
	twisted=SpatialGridPushPull((70,70,70),(lambda X:twist(X,0.01),lambda I:twist(I,-0.01)))
	
	# Add some scalar data (a ball centered on (50,50,50) with radius 10)
	X=twisted.getCoordinates()
	twisted.dat['mydata']=(np.sqrt((X[0]-50)**2+(X[1]-50)**2+(X[2]-50)**2)<10).astype('float')

	# Plot everything
	p = pv.Plotter(window_size=[2000, 2000])
	grid=twisted.toPyvista()
	p.add_mesh(grid,style='wireframe', color="red",opacity=0.1)
	p.add_mesh(grid.contour([0.5],grid.point_data['mydata']), color="red",copy_mesh=True)
	p.show_grid()
	p.show()

def testDataTransfer():
	# Create 2 simple grids, with Identity push and pull functions (and a slight shift)
	sg1=SpatialGridAffine((100,100,100),np.eye(4))
	sg2=SpatialGridPushPull((50,50,50),(lambda I : shift(I,(0.5,0.5,0.5)),lambda X : shift(X,(-0.5,-0.5,-0.5))))

	# Add some scalar data (a ball centered on (50,50,50) with radius 30) on the first grid
	X=sg1.getCoordinates()
	sg1.dat['mydata']=(np.sqrt((X[0]-50)**2+(X[1]-50)**2+(X[2]-50)**2)<30).astype('float')

	#Transfer the data onto the second grid
	sg2.transferScalarData(sg1)

	# Plot everything
	p = pv.Plotter(window_size=[2000, 2000])
	grid1=sg1.toPyvista()
	grid2=sg2.toPyvista()
	p.add_mesh(grid1,style='wireframe', color="red",opacity=0.1)
	p.add_mesh(grid2,style='wireframe', color="blue",opacity=0.1)
	p.add_mesh(grid1.contour([0.5],grid1.point_data['mydata']), color="red",copy_mesh=True)
	p.add_mesh(grid2.contour([0.5],grid2.point_data['mydata']), color="blue",copy_mesh=True)
	p.show_grid()
	p.show()

def testAffineGrid():
	affinegrid=SpatialGridAffine((100,100,100),np.array([[1,-1,0,0],[1,1,1,0],[1,0,-1,0],[0,0,0,1]]))
	X=affinegrid.getCoordinates()
	Xe=affinegrid.getExtent()
	mid=[0.5*(Xe[i][0]+Xe[i][1]) for i in range(3)]
	affinegrid.dat['mydata']=((np.abs(X[0]-mid[0])+np.abs(X[1]-mid[1])+np.abs(X[2]-mid[2]))<40).astype('float')


	# Plot everything
	p = pv.Plotter(window_size=[2000, 2000])

	grid3=affinegrid.toPyvista()
	p.add_mesh(grid3,style='wireframe', color="green",opacity=0.1)
	p.add_mesh(grid3.contour([0.5],grid3.point_data['mydata']), color="green",copy_mesh=True)

	p.show_grid()
	p.show()

def testRescale():
	affinegrid=SpatialGridAffine((100,100,100),np.array([[1,-1,0,0],[1,1,1,0],[1,0,-1,0],[0,0,0,1]]))
	X=affinegrid.getCoordinates()
	Xe=affinegrid.getExtent()
	mid=[0.5*(Xe[i][0]+Xe[i][1]) for i in range(3)]
	affinegrid.dat['mydata']=((np.abs(X[0]-mid[0])+np.abs(X[1]-mid[1])+np.abs(X[2]-mid[2]))<40).astype('float')
	
	affinegrid_reduced=affinegrid.copy()
	affinegrid_reduced.rescale((50,50,50))


	print(affinegrid)
	print(affinegrid_reduced)


	# Plot everything
	p = pv.Plotter(window_size=[2000, 2000])

	grid3=affinegrid.toPyvista()
	p.add_mesh(grid3,style='wireframe', color="green",opacity=0.1)
	p.add_mesh(grid3.contour([0.5],grid3.point_data['mydata']), color="green",copy_mesh=True)

	grid4=affinegrid_reduced.toPyvista()
	p.add_mesh(grid4,style='wireframe', color="yellow",opacity=0.1)
	p.add_mesh(grid4.contour([0.5],grid4.point_data['mydata']), color="yellow",copy_mesh=True)

	p.show_grid()
	p.show()

def testPerspective():

	perspectivegrid=SpatialGridPerspective((100,100,100),focal=(-30,0,0),forward=(1,1,1),up=(1,1,-2),left=(1,-1,0),angle_up=45,angle_down=-10,angle_left=10,angle_right=10,start=1,end=100)
	perspectivegrid2=SpatialGridPerspective((100,100,100),focal=(-30,0,0),forward=(1,1,1),up=(1,1,-2),left=(1,-1,0),angle_up=10,angle_down=10,angle_left=10,angle_right=10,start=1,end=100)
	grid5=perspectivegrid.toPyvista()
	grid6=perspectivegrid2.toPyvista()
	p = pv.Plotter(window_size=[2000, 2000])
	p.add_mesh(grid5,style='wireframe', color="black",opacity=0.1)
	p.add_mesh(grid6,style='wireframe', color="red",opacity=0.1)
	p.show_grid()
	p.show()

def testImage():

	Mat=np.array([[1,1,0],[1,-1,0],[1,0,0],[0,0,1]])
	imagegrid=SpatialGridAffine((100,100),affine=Mat)
	I=imagegrid.getIndices()
	imagegrid.dat['mydata']=(np.sqrt((I[0]-50)**2+(I[1]-50)**2)<30).astype('float')


	p = pv.Plotter(window_size=[2000, 2000])
	grid=imagegrid.toPyvista()
	p.add_mesh(grid,style='wireframe', color="black",opacity=0.1)
	p.add_mesh(grid,cmap='Reds',clim=[0,1],opacity='linear')

	p.show_grid()
	p.show()

def testSquish():

	straight=SpatialGridAffine((100,100,100),affine=np.eye(4),dat={'dull':np.ones((100,100,100))})
	print(straight)

	perspectivegrid=SpatialGridPerspective((100,100,100),angle_left=30,angle_up=45,start=50,end=100)
	X=perspectivegrid.getCoordinates()
	perspectivegrid.dat['mydata']=(np.sqrt((X[0]-75)**2+X[1]**2+X[2]**2)<10).astype('float')

	squishedgrid=perspectivegrid.squish(150)

	print(perspectivegrid)
	print(squishedgrid)

	grid5=perspectivegrid.toPyvista()
	grid6=squishedgrid.toPyvista()

	p = pv.Plotter(window_size=[2000, 2000])
	p.add_mesh(grid5,style='wireframe', color="black",opacity=0.1)
	p.add_mesh(grid5.contour([0.5],grid5.point_data['mydata']), color="black",copy_mesh=True)

	p.add_mesh(grid6,style='wireframe', color="red",opacity=0.1)
	p.add_mesh(grid6,cmap='Reds',opacity='linear')
	p.show_grid()
	p.show()


def testSparseGrid():

	# Create a simple grid, with Identity push and pull functions
	straight=SpatialGridAffine((100,100,100),np.eye(4))

	# Add some scalar data (a ball centered on (20,50,30) with radius 10)
	X=straight.getCoordinates()

	straight.dat['mydata']=(np.sqrt((X[0]-20)**2+(X[1]-50)**2+(X[2]-30)**2)<10).astype('float')

	sparse=RBFsparseDat(straight,'mydata',0.05,0.5)

	print(sparse.sparse_dat.min(),sparse.sparse_dat.max())
	print(sparse.RBFWeights.min(),sparse.RBFWeights.max())

	straight.dat['mydata_RBF']=listdat_to_dat(sparse.eval(X_to_listpoints(straight.getCoordinates())),straight.shape)


	# Print the object representation
	print(straight)

	# Plot everything
	p = pv.Plotter(window_size=[2000, 2000])

	grid=straight.toPyvista()
	p.add_mesh(grid,style='wireframe', color="red",opacity=0.1)
	p.add_mesh(grid.contour([0.5],grid.point_data['mydata']), color="red",copy_mesh=True,opacity=0.5)
	p.add_mesh(grid.contour([0.5],grid.point_data['mydata_RBF']), color="blue",copy_mesh=True,opacity=0.5)
	p.show_grid()
	p.show()



def testBundle():

	# Create a simple grid, with Identity push and pull functions
	straight=SpatialGridAffine((100,100,100),np.eye(4))

	# Add some scalar data (a ball centered on (20,50,30) with radius 10)
	X=straight.getCoordinates()

	straight.dat['mydata']=(np.sqrt((X[0]-20)**2+(X[1]-50)**2+(X[2]-30)**2)<10).astype('float')

	sparse=RBFsparseDat(straight,'mydata',0.05,0.5)

	myBundle=Bundle([50,50,50],sparse.points,np.pi/5)
	Mat=myBundle.distPointsMatrix(sparse.points)
	print(Mat.min(),Mat.max())





if __name__=='__main__':
	#testSimpleGrid()
	#testTwistedGrid()
	#testDataTransfer()
	#testAffineGrid()
	#testRescale()
	#testPerspective()
	#testImage()
	#testSquish()
	#testSparseGrid()
	testBundle()
