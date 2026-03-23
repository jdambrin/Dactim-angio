import numpy as np
from scipy.spatial import KDTree
from scipy.special import erf
import matplotlib.pyplot as plt
import warnings

def opposite_correlation(x,y):
	sx=np.sqrt(np.sum(x**2))
	sy=np.sqrt(np.sum(y**2))
	corr=np.sum(x*y)/(sx*sy)
	dycorr=x/(sx*sy)-np.sum(x*y)*y/(sx*(sy**3))
	dxcorr=y/(sx*sy)-np.sum(x*y)*x/(sy*(sx**3))
	return -corr,-dxcorr,-dycorr

def blabla(x,y):
	return np.sum(x**2)+np.sum(y**2),2*x,2*y

def opposite_mutual_information_kernel(x,y,bins=50,plot_hist=False):

	warnings.filterwarnings("ignore")

	scale_x=x.max()
	scale_y=y.max()

	x=x/scale_x
	y=y/scale_y

	
	M=len(x)

	e=np.linspace(0-5/bins,1+5/bins,bins)
	N=e.shape[0]-1

	sigma=np.min(e[1:]-e[:-1])
	[x_grid,y_grid]=np.meshgrid(0.5*(e[:-1]+e[1:]),0.5*(e[:-1]+e[1:]))

	hgram=np.zeros((N,N))

	Tree=KDTree(np.array([x,y]).T)

	Nei=Tree.query_ball_point(np.array([x_grid.ravel(),y_grid.ravel()]).T,r=5*sigma)

	for j in range(N):
		for i in range(N):
			k=i+N*j
			hgram[i,j]=0.125*(sigma/M)*np.sum(
				(erf((e[i+1]-x[Nei[k]])/sigma)-erf((e[i]-x[Nei[k]])/sigma))*
				(erf((e[j+1]-y[Nei[k]])/sigma)-erf((e[j]-y[Nei[k]])/sigma))
				)

	pxy = hgram#/float(np.sum(hgram))

	px = np.sum(pxy, axis=1) # marginal for x over y
	py = np.sum(pxy, axis=0) # marginal for y over x
	px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals

	e22=e[:-1, None]*np.ones(len(e))[None,:-1]
	e11=np.ones(len(e))[:-1,None]*e[None,:-1]

	e22p=e[1:, None]*np.ones(len(e))[None,:-1]
	e11p=np.ones(len(e))[:-1,None]*e[None,1:]


	#plt.contourf(x_grid,y_grid,e11.T)
	#plt.scatter(x,y)
	#plt.show()

	nzs = pxy > 0 # Only non-zero pxy values contribute to the sum

	DU_I=np.zeros(M)
	DV_I=np.zeros(M)

	def expsq(t):
		return np.exp(-t**2)

	nzs_idx=list(np.where(nzs)[0]+N*np.where(nzs)[1])
	nzs_idx=np.sort(nzs_idx)

	e11_ravel=e11.ravel()
	e22_ravel=e22.ravel()
	e11p_ravel=e11p.ravel()
	e22p_ravel=e22p.ravel()

	#Dp_I_ravel=np.zeros(M**2)
	#Dp_I_ravel[nzs_idx]=(np.log(pxy[nzs].T)-np.log(px_py[nzs].T)-1).ravel()
	Dp_I_ravel = (np.log(pxy.T)-np.log(px_py.T)-1).ravel()

	Tree2=KDTree(np.array([x_grid.ravel(),y_grid.ravel()]).T)

	Nei2=Tree2.query_ball_point(np.array([x,y]).T,r=5*sigma,return_sorted=True)

	for k in range(M):
		indices= Nei2[k]
		DU_I[k]=-0.25*(1/(np.sqrt(np.pi)*M))*np.sum(
			Dp_I_ravel[indices]*
			(expsq((e11p_ravel[indices]-x[k])/sigma)-expsq((e11_ravel[indices]-x[k])/sigma))*
			(erf((e22p_ravel[indices]-y[k])/sigma)-erf((e22_ravel[indices]-y[k])/sigma))
			)
		DV_I[k]=-0.25*(1/(np.sqrt(np.pi)*M))*np.sum(
			Dp_I_ravel[indices]*
			(expsq((e22p_ravel[indices]-y[k])/sigma)-expsq((e22_ravel[indices]-y[k])/sigma))*
			(erf((e11p_ravel[indices]-x[k])/sigma)-erf((e11_ravel[indices]-x[k])/sigma))
			) 

	I=np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

	if plot_hist:
		import matplotlib.pyplot as plt
		plt.contourf(x_grid,y_grid,np.log(hgram).T)
		plt.scatter(x,y)
		plt.quiver(x,y,-DU_I,-DV_I)
		plt.xlim([-0.1,1.1])
		plt.ylim([-0.1,1.1])
		plt.show()

	return -I,-DU_I/scale_x,-DV_I/scale_y


def joint_entropy_kraskov(x,y,k=5,epsilon=0.001):
	x=x.flatten()
	y=y.flatten()
	Points=np.array([x,y]).T
	tree=KDTree(Points)
	d,i=tree.query(Points,k)
	E=np.sum(np.log(d[:,1:]+epsilon))
	grad_x_E=np.sum((np.repeat(np.array([x]).T,k-1,axis=1)-x[i[:,1:]])*(d[:,1:]+epsilon)**(-2),axis=1)
	grad_y_E=np.sum((np.repeat(np.array([y]).T,k-1,axis=1)-y[i[:,1:]])*(d[:,1:]+epsilon)**(-2),axis=1)
	return E,grad_x_E,grad_y_E


def mutual_info_kraskov(x,y):
	pass



def singular_exclusion(x,y):
	x=x.flatten()
	y=y.flatten()
	si=np.sum(np.exp(-(x**2)/0.1)+np.exp(-(y**2)/0.1))
	si_x=-(2*x/0.1)*np.exp(-(x**2)/0.1)
	si_y=-(2*y/0.1)*np.exp(-(y**2)/0.1)
	return si, si_x, si_y

def mse(x,y):
	return np.sum((x-y)**2),2*(x-y),2*(y-x)


if __name__ == "__main__":
	x=np.random.rand(1000)
	y=np.random.rand(1000)
	h=0.001
	for i in range(10000):
		E,dxE,dyE=joint_entropy_kraskov(x,y,[10])
		h=0.01/max(np.abs(dxE).max(),np.abs(dyE).max())
		x=x-h*dxE
		y=y-h*dyE
		plt.scatter(x,y)
		plt.quiver(x,y,dxE,dyE)
		plt.show()


