import numpy as np


def opposite_correlation(x,y):
	sx=np.sqrt(np.sum(x**2))
	sy=np.sqrt(np.sum(y**2))
	corr=np.sum(x*y)/(sx*sy)
	dycorr=x/(sx*sy)-np.sum(x*y)*y/(sx*(sy**3))
	return -corr,-dycorr

def blabla(x,y):
	return np.sum(y),0*y+1

def mutual_information_kernel(im1,im2,hgram=None,bins=50):
	max1=np.max(im1)
	max2=np.max(im2)
	min1=np.min(im1)
	min2=np.min(im2)

	if(min1<max1):
		X1=((im1-min1)/(max1-min1)).flatten()
	else:
		X1=(0*im1).flatten()
	if(min2<max2):
		X2=((im2-min2)/(max2-min2)).flatten()
	else:
		X2=(0*im2).flatten()

	e=np.linspace(0-5/bins,1+5/bins,bins)
	N=e.shape[0]-1

	sigma=np.min(e[1:]-e[:-1])
	[X1_grid,X2_grid]=np.meshgrid(0.5*(e[:-1]+e[1:]),0.5*(e[:-1]+e[1:]))

	if(type(hgram)==type(None)):
		hgram=np.zeros((N,N))
		Tree=KDTree(np.array([X1,X2]).T)
		Nei=Tree.query_ball_point(np.array([X1_grid.ravel(),X2_grid.ravel()]).T,r=5*sigma)
		for j in range(N):
			for i in range(N):
				k=i+N*j
				hgram[i,j]=0.25*(1/len(X1.ravel()))*np.sum(
					(erf((e[i+1]-X1[Nei[k]])/sigma)-erf((e[i]-X1[Nei[k]])/sigma))*
					(erf((e[j+1]-X2[Nei[k]])/sigma)-erf((e[j]-X2[Nei[k]])/sigma))
					)

	pxy = hgram#/float(np.sum(hgram))
	px = np.sum(pxy, axis=1) # marginal for x over y
	py = np.sum(pxy, axis=0) # marginal for y over x
	px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
	p_xx=px[:, None]*np.ones(len(py))[None,:]
	p_yy=np.ones(len(px))[:,None]*py[None,:]

	e22=e[:-1, None]*np.ones(len(e))[None,:-1]
	e11=np.ones(len(e))[:-1,None]*e[None,:-1]

	e22p=e[1:, None]*np.ones(len(e))[None,:-1]
	e11p=np.ones(len(e))[:-1,None]*e[None,1:]

	nzs = pxy > 0 # Only non-zero pxy values contribute to the sum

	M=len(im1.ravel())
	DU_I=np.zeros(M)
	Dp_I=np.log(pxy[nzs]/px_py[nzs])-1

	def expsq(t):
		return np.exp(-t**2)

	nzs_idx=list(np.where(nzs)[0]+N*np.where(nzs)[1])
	nzs_idx=np.sort(nzs_idx)

	e11_ravel=e11.ravel()
	e22_ravel=e22.ravel()
	e11p_ravel=e11p.ravel()
	e22p_ravel=e22p.ravel()
	Dp_I_ravel=(np.log(pxy.T/px_py.T)-1).ravel()
	Tree2=KDTree(np.array([X1_grid.ravel(),X2_grid.ravel()]).T)
	Nei2=Tree2.query_ball_point(np.array([X1,X2]).T,r=5*sigma,return_sorted=True)
	for k in range(M):
		indices=Nei2[k]
		DU_I[k]=-0.5*(1/(np.sqrt(np.pi)*sigma*M))*np.sum(
			Dp_I_ravel[indices]*
			(expsq((e11p_ravel[indices]-X1[k])/sigma)-expsq((e11_ravel[indices]-X1[k])/sigma))*
			(erf((e22p_ravel[indices]-X2[k])/sigma)-erf((e22_ravel[indices]-X2[k])/sigma))
			) 

	I=np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
	return I,DU_I.reshape(im1.shape),hgram


def joint_entropy_kraskov(x,y):
	pass

def mutual_info_kraskov(x,y):
	pass

#def singular_exclusion(x,y):
#	si=np.exp(-x^2/sx)+
