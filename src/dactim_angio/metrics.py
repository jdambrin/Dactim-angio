import numpy as np
from scipy.spatial import KDTree
from scipy.special import erf
import matplotlib.pyplot as plt

def opposite_correlation(x,y,args=[]):
	sx=np.sqrt(np.sum(x**2))
	sy=np.sqrt(np.sum(y**2))
	corr=np.sum(x*y)/(sx*sy)
	dycorr=x/(sx*sy)-np.sum(x*y)*y/(sx*(sy**3))
	dxcorr=y/(sx*sy)-np.sum(x*y)*x/(sy*(sx**3))
	return -corr,-dxcorr,-dycorr

def blabla(x,y,args=[]):
	return np.sum(x**2)+np.sum(y**2),2*x,2*y

def mutual_information_kernel(x,y,args=[(0,1),(0,1),50]):
	bounds_x=args[0]
	bounds_y=args[1]
	bins=args[2]

	print(bounds_x,bounds_y)
	
	x=(x-bounds_x[0])/(bounds_x[1]-bounds_x[0])
	y=(y-bounds_y[0])/(bounds_y[1]-bounds_y[0])
	
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

	#plt.contourf(x_grid,y_grid,np.log(hgram).T)
	#plt.scatter(x,y)
	#plt.show()


	px = np.sum(pxy, axis=1) # marginal for x over y
	py = np.sum(pxy, axis=0) # marginal for y over x
	px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
	p_xx=px[:, None]*np.ones(len(py))[None,:]
	p_yy=np.ones(len(px))[:,None]*py[None,:]

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
	Dp_I=np.log(pxy[nzs]/px_py[nzs])-1

	def expsq(t):
		return np.exp(-t**2)

	nzs_idx=list(np.where(nzs)[0]+N*np.where(nzs)[1])
	nzs_idx=np.sort(nzs_idx)

	e11_ravel=e11.ravel()
	e22_ravel=e22.ravel()
	e11p_ravel=e11p.ravel()
	e22p_ravel=e22p.ravel()
	Dp_I_ravel=(np.log(pxy.T)-np.log(px_py.T)-1).ravel()

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

	return I,DU_I/(bounds_x[1]-bounds_x[0]),DV_I/(bounds_y[1]-bounds_y[0])


def joint_entropy_kraskov(x,y):
	pass

def mutual_info_kraskov(x,y):
	pass


from scipy.optimize import check_grad

if __name__ == '__main__':

	x0=np.random.rand(100)
	y0=np.random.rand(100)

	func_to_test=mutual_information_kernel

	def func_x(x):
		return func_to_test(x,y0)[0]
	def grad_func_x(x):
		return func_to_test(x,y0)[1]

	def func_y(y):
		return func_to_test(x0,y)[0]
	def grad_func_y(y):
		return func_to_test(x0,y)[2]


	epsilons=[10**(-i) for i in range(3,11)]
	err_grad_x=[]
	for eps in epsilons:
		err_grad_x.append(check_grad(func_x,grad_func_x,x0,epsilon=eps))

	err_grad_y=[]
	for eps in epsilons:
		err_grad_y.append(check_grad(func_y,grad_func_y,y0,epsilon=eps))

	plt.loglog(epsilons,err_grad_x,'.-')
	plt.loglog(epsilons,err_grad_y,'.-')

	plt.show()


#def singular_exclusion(x,y):
#	si=np.exp(-x^2/sx)+
