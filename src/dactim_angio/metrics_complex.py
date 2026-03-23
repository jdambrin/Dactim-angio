import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import check_grad
from scipy.spatial import KDTree
from scipy.special import digamma
from scipy.special import gamma
from scipy.special import erf
import time

def mutual_information_kraskov_stoegbauer_grassberger(x,y,k=3,method=1):

	N=x.shape[0]
	d=x.shape[1]
	xy=np.concatenate((x,y),axis=1)
	tree_xy=KDTree(xy)
	
	foo,i=tree_xy.query(xy,k=k+1,p=np.inf)
	
	idx=i[:,-1]

	x_nei=xy[idx,:d]
	y_nei=xy[idx,d:]

	
	dists_xy=np.linalg.norm(xy-xy[idx,:],axis=1,ord=np.inf)

	if(method==1):
		dists_x=dists_xy
		dists_y=dists_xy
	elif(method==2):
		dists_x=np.linalg.norm(x-x_nei,axis=1,ord=np.inf)
		dists_y=np.linalg.norm(y-y_nei,axis=1,ord=np.inf)



	tree_x=KDTree(x)
	tree_y=KDTree(y)


	nx=[len(list_nei)-1 for list_nei in tree_x.query_ball_point(x, r=dists_x, p=np.inf)]
	ny=[len(list_nei)-1 for list_nei in tree_y.query_ball_point(y, r=dists_y, p=np.inf)]

	print('nx=',nx)
	print('ny=',ny)
	

	Mi=digamma(k)+digamma(N)-(1/N)*np.sum(digamma(nx)+digamma(ny))

	return Mi,0,0

def mutual_information_kozachenko_leonenko(x,y,k=1,eps=0):

	xy=np.concatenate((x,y),axis=1)
	
	Ent_xy,dxy_Ent_xy=entropy_kozachenko_leonenko(xy,k=k,eps=eps)
	Ent_x ,dx_Ent_x  =entropy_kozachenko_leonenko(x,k=k,eps=eps)
	Ent_y ,dy_Ent_y  =entropy_kozachenko_leonenko(y,k=k,eps=eps)

	dim=x.shape[1]

	Mi=Ent_x+Ent_y-Ent_xy

	print(Ent_x,Ent_y,Ent_xy)

	dx_Mi=dx_Ent_x-dxy_Ent_xy[:,:dim]
	dy_Mi=dy_Ent_y-dxy_Ent_xy[:,dim:]

	return Mi,dx_Mi,dy_Mi


def mutual_information_kernel(x,y,idx_sample=[],bandwidth=0.1):

	N=x.shape[0]
	d=x.shape[1]

	if(len(idx_sample)==0):
		idx_sample=np.arange(N)

	def K(x) : 
		return (1/np.sqrt(2*np.pi))*np.exp(-0.5*(x**2))

	#if(idx_sample==None):
	#	idx_sample=np.arange(N)

	Nsample=len(idx_sample)

	xy=np.concatenate((x,y),axis=1)

	xy_sample=xy[idx_sample,:]
	x_sample=x[idx_sample,:]
	y_sample=y[idx_sample,:]

	tree_xy=KDTree(xy)
	tree_x=KDTree(x)
	tree_y=KDTree(y)

	idx_neigh_xy_sample=tree_xy.query_ball_point(xy_sample,5*bandwidth,return_sorted=True)
	idx_neigh_x_sample=tree_x.query_ball_point(x_sample,5*bandwidth,return_sorted=True)
	idx_neigh_y_sample=tree_y.query_ball_point(y_sample,5*bandwidth,return_sorted=True)


	Mi=0
	dxy_Mi=np.zeros((N,2*d))

	for i in range(Nsample):

		pxy_i = (1/(N*(bandwidth**(2*d))))*np.sum(K(np.linalg.norm(xy_sample[i,:]-xy[idx_neigh_xy_sample[i],:],axis=1)/bandwidth))
		px_i  = (1/(N*(bandwidth**d)))*np.sum(K(np.linalg.norm(x_sample[i,:]-x[idx_neigh_x_sample[i],:],axis=1)/bandwidth))
		py_i  = (1/(N*(bandwidth**d)))*np.sum(K(np.linalg.norm(y_sample[i,:]-y[idx_neigh_y_sample[i],:],axis=1)/bandwidth))

		dxy_pxy_i=np.zeros((N,2*d))
		dxy_px_i=np.zeros((N,2*d))
		dxy_py_i=np.zeros((N,2*d))


		dxy_pxy_i[idx_neigh_xy_sample[i],:] = (1/(N*(bandwidth**(2*d+2))))*(xy_sample[i,:]-xy[idx_neigh_xy_sample[i],:])*K(np.linalg.norm(xy_sample[i,:]-xy[idx_neigh_xy_sample[i],:],axis=1,keepdims=True)/bandwidth)
		dxy_pxy_i[idx_sample[i],:]=-(1/(N*(bandwidth**(2*d+2))))*np.sum((xy_sample[i,:]-xy[idx_neigh_xy_sample[i],:])*K(np.linalg.norm(xy_sample[i,:]-xy[idx_neigh_xy_sample[i],:],axis=1,keepdims=True)/bandwidth),axis=0,keepdims=True)

		dxy_px_i[idx_neigh_x_sample[i],:d] = (1/(N*(bandwidth**(d+2))))*(x_sample[i,:]-x[idx_neigh_x_sample[i],:])*K(np.linalg.norm(x_sample[i,:]-x[idx_neigh_x_sample[i],:],axis=1,keepdims=True)/bandwidth)
		dxy_px_i[idx_sample[i],:d]=-(1/(N*(bandwidth**(d+2))))*np.sum((x_sample[i,:]-x[idx_neigh_x_sample[i],:])*K(np.linalg.norm(x_sample[i,:]-x[idx_neigh_x_sample[i],:],axis=1,keepdims=True)/bandwidth),axis=0,keepdims=True)

		dxy_py_i[idx_neigh_y_sample[i],d:] = (1/(N*(bandwidth**(d+2))))*(y_sample[i,:]-y[idx_neigh_y_sample[i],:])*K(np.linalg.norm(y_sample[i,:]-y[idx_neigh_y_sample[i],:],axis=1,keepdims=True)/bandwidth)
		dxy_py_i[idx_sample[i],d:]=-(1/(N*(bandwidth**(d+2))))*np.sum((y_sample[i,:]-y[idx_neigh_y_sample[i],:])*K(np.linalg.norm(y_sample[i,:]-y[idx_neigh_y_sample[i],:],axis=1,keepdims=True)/bandwidth),axis=0,keepdims=True)


		Mi += np.log(pxy_i/(px_i*py_i))

		dxy_Mi += dxy_pxy_i/pxy_i - (dxy_px_i/px_i+dxy_py_i/py_i)


	Mi/=Nsample
	dxy_Mi/=Nsample

	dx_Mi=dxy_Mi[:,:d]
	dy_Mi=dxy_Mi[:,d:]
	
	return Mi,dx_Mi,dy_Mi



def angle(x,y):
	return(np.sum(x*y),y,x)



def mutual_information_kernel_vectorized(x, y, idx_sample=None, bandwidth=0.1):
    """
    Version vectorisée de mutual_information_kernel.
    On évite la boucle Python sur i en manipulant tous les couples (i, voisin) d'un coup.
    """
    N, d = x.shape
    if idx_sample is None or len(idx_sample) == 0:
        idx_sample = np.arange(N, dtype=int)
    else:
        idx_sample = np.asarray(idx_sample, dtype=int)

    Nsample = idx_sample.shape[0]

    def K(z):
        return (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (z ** 2))

    xy = np.concatenate((x, y), axis=1)
    xy_sample = xy[idx_sample, :]
    x_sample = x[idx_sample, :]
    y_sample = y[idx_sample, :]

    # KDTree
    tree_xy = KDTree(xy)
    tree_x = KDTree(x)
    tree_y = KDTree(y)

    # Listes de voisins (ragged lists)
    idx_neigh_xy_sample=tree_xy.query_ball_point(xy_sample,5*bandwidth,return_sorted=True)
    idx_neigh_x_sample=tree_x.query_ball_point(x_sample,5*bandwidth,return_sorted=True)
    idx_neigh_y_sample=tree_y.query_ball_point(y_sample,5*bandwidth,return_sorted=True)


    # --- Utilitaires pour transformer listes de voisins -> (I,J) vectorisés ---
    def build_pairs(neigh_list):
        """
        neigh_list : liste de longueur Nsample, élément i = np.array des indices voisins dans [0..N-1]
        Retourne :
          I : indices des points "i" (dans [0..Nsample-1])
          J : indices des voisins (dans [0..N-1])
          counts : nombre de voisins par i
        """
        counts = np.fromiter((len(idx) for idx in neigh_list), int, len(neigh_list))
        I = np.repeat(np.arange(len(neigh_list)), counts)
        J = np.concatenate(neigh_list) if len(neigh_list) > 0 else np.array([], dtype=int)
        return I, J, counts

    I_xy, J_xy, _ = build_pairs(idx_neigh_xy_sample)
    I_x,  J_x,  _ = build_pairs(idx_neigh_x_sample)
    I_y,  J_y,  _ = build_pairs(idx_neigh_y_sample)

    h = bandwidth

    # --- Densités p(x,y), p(x), p(y) sur les points échantillons ---

    # p(x,y)
    diffs_xy = xy_sample[I_xy] - xy[J_xy]        # (M_xy, 2d)
    norms_xy = np.linalg.norm(diffs_xy, axis=1)  # (M_xy,)
    K_xy = K(norms_xy / h)

    pxy = np.zeros(Nsample, dtype=float)
    factor_xy = 1.0 / (N * (h ** (2 * d)))
    np.add.at(pxy, I_xy, K_xy)
    pxy *= factor_xy

    # p(x)
    diffs_x = x_sample[I_x] - x[J_x]
    norms_x = np.linalg.norm(diffs_x, axis=1)
    K_x = K(norms_x / h)

    px = np.zeros(Nsample, dtype=float)
    factor_x = 1.0 / (N * (h ** d))
    np.add.at(px, I_x, K_x)
    px *= factor_x

    # p(y)
    diffs_y = y_sample[I_y] - y[J_y]
    norms_y = np.linalg.norm(diffs_y, axis=1)
    K_y = K(norms_y / h)

    py = np.zeros(Nsample, dtype=float)
    factor_y = 1.0 / (N * (h ** d))
    np.add.at(py, I_y, K_y)
    py *= factor_y

    # --- Mutual information ---
    Mi = np.mean(np.log(pxy / (px * py)))

    # --- Gradients dI/dx, dI/dy (taille N x d) ---
    dx_Mi = np.zeros((N, d), dtype=float)
    dy_Mi = np.zeros((N, d), dtype=float)

    # 1) Terme joint p(x,y) : dI += d(pxy)/pxy
    factor_grad_xy = 1.0 / (N * (h ** (2 * d + 2)))
    common_xy = factor_grad_xy * (K_xy / pxy[I_xy])  # (M_xy,)

    # contributions pour parties x et y (on découpe xy = [x, y])
    contrib_x_xy = common_xy[:, None] * (x_sample[I_xy] - x[J_xy])  # (M_xy, d)
    contrib_y_xy = common_xy[:, None] * (y_sample[I_xy] - y[J_xy])  # (M_xy, d)

    # accumulation sur les voisins j
    np.add.at(dx_Mi, J_xy, contrib_x_xy)
    np.add.at(dy_Mi, J_xy, contrib_y_xy)

    # termes négatifs sur les points échantillons idx_sample[i]
    tmp_sum_dx_sample = np.zeros((Nsample, d), dtype=float)
    tmp_sum_dy_sample = np.zeros((Nsample, d), dtype=float)

    np.add.at(tmp_sum_dx_sample, I_xy, contrib_x_xy)
    np.add.at(tmp_sum_dy_sample, I_xy, contrib_y_xy)
    np.add.at(dx_Mi, idx_sample, -tmp_sum_dx_sample)
    np.add.at(dy_Mi, idx_sample, -tmp_sum_dy_sample)

    # 2) Terme marginal p(x) : dI -= d(px)/px
    factor_grad_x = 1.0 / (N * (h ** (d + 2)))
    common_x = factor_grad_x * (K_x / px[I_x])

    pair_contrib_dx = common_x[:, None] * (x_sample[I_x] - x[J_x])  # (M_x, d)

    # voisins j : signe négatif
    np.add.at(dx_Mi, J_x, -pair_contrib_dx)

    # contributions sur idx_sample[i] (signe positif)
    tmp_sum_dx_sample = np.zeros((Nsample, d), dtype=float)
    np.add.at(tmp_sum_dx_sample, I_x, pair_contrib_dx)
    np.add.at(dx_Mi, idx_sample, tmp_sum_dx_sample)

    # 3) Terme marginal p(y) : dI -= d(py)/py
    factor_grad_y = 1.0 / (N * (h ** (d + 2)))
    common_y = factor_grad_y * (K_y / py[I_y])

    pair_contrib_dy = common_y[:, None] * (y_sample[I_y] - y[J_y])  # (M_y, d)

    # voisins j : signe négatif
    np.add.at(dy_Mi, J_y, -pair_contrib_dy)

    # contributions sur idx_sample[i] (signe positif)
    tmp_sum_dy_sample = np.zeros((Nsample, d), dtype=float)
    np.add.at(tmp_sum_dy_sample, I_y, pair_contrib_dy)
    np.add.at(dy_Mi, idx_sample, tmp_sum_dy_sample)

    # Normalisation finale par Nsample
    dx_Mi /= Nsample
    dy_Mi /= Nsample

    return Mi, dx_Mi, dy_Mi



def mutual_information_kernel_grid(x,y,Ngrid=30):

	N=x.shape[0]

	d=x.shape[1]

	def K(x) : 
		return (1/np.sqrt(2*np.pi))*np.exp(-0.5*(x**2))

	xy=np.concatenate((x,y),axis=1)

	N_out=5

	h=2/(Ngrid-1)
	bandwidth=h

	N_out=int(np.ceil(5*bandwidth/h))
	minval=-1-N_out*h
	maxval=1+N_out*h

	#print(np.linspace(minval,maxval,Ngrid+2*N_out))


	xy_grid=np.reshape(np.array(np.meshgrid(*[np.linspace(minval,maxval,Ngrid+2*N_out) for _ in range(2*d)])),(2*d,(Ngrid+2*N_out)**(2*d))).T
	x_grid=xy_grid[:,:d]
	y_grid=xy_grid[:,d:]

	tree_xy=KDTree(xy)
	tree_x=KDTree(x)
	tree_y=KDTree(y)

	idx_neigh_xy_grid = tree_xy.query_ball_point(xy_grid,5*bandwidth,p=np.inf)
	idx_neigh_x_grid  = tree_x.query_ball_point(x_grid,5*bandwidth,p=np.inf)
	idx_neigh_y_grid  = tree_y.query_ball_point(y_grid,5*bandwidth,p=np.inf)

	reduce_idx=np.array([i for i, lst in enumerate(idx_neigh_xy_grid) if lst],dtype=int)

	Mi=0

	#print(xy_grid)

	#histo=np.zeros(tuple([Ngrid+2*N_out for _ in range(2*d)]))

	dxy_Mi=np.zeros((N,2*d))

	print(len(reduce_idx),(Ngrid+2*N_out)**(2*d))

	for i in reduce_idx:
		'''
		# A virer
		pxy_i=(1/N)*np.sum(np.prod(
			erf((xy_grid[i,:]+0.5*h-xy[idx_neigh_xy_grid[i],:])/bandwidth)-erf((xy_grid[i,:]-0.5*h-xy[idx_neigh_xy_grid[i],:])/bandwidth)
			,axis=1))

		px_i=(1/N)*np.sum(np.prod(
			erf((x_grid[i,:]+0.5*h-x[idx_neigh_x_grid[i],:])/bandwidth)-erf((x_grid[i,:]-0.5*h-x[idx_neigh_x_grid[i],:])/bandwidth)
			,axis=1)*h*h)

		py_i=(1/N)*np.sum(np.prod(
			erf((y_grid[i,:]+0.5*h-y[idx_neigh_y_grid[i],:])/bandwidth)-erf((y_grid[i,:]-0.5*h-y[idx_neigh_y_grid[i],:])/bandwidth)
			,axis=1)*h*h)
		'''

		pxy_i = (1/(N*(bandwidth**(2*d))))*np.sum(K(np.linalg.norm(xy_grid[i,:]-xy[idx_neigh_xy_grid[i],:],axis=1)/bandwidth))
		px_i  = (1/(N*(bandwidth**d)))*np.sum(K(np.linalg.norm(x_grid[i,:]-x[idx_neigh_x_grid[i],:],axis=1)/bandwidth))
		py_i  = (1/(N*(bandwidth**d)))*np.sum(K(np.linalg.norm(y_grid[i,:]-y[idx_neigh_y_grid[i],:],axis=1)/bandwidth))

		dxy_pxy_i=np.zeros((N,2*d))
		dxy_px_i=np.zeros((N,2*d))
		dxy_py_i=np.zeros((N,2*d))

		dxy_pxy_i[idx_neigh_xy_grid[i],:] = (1/(N*(bandwidth**(2*d+2))))*(xy_grid[i,:]-xy[idx_neigh_xy_grid[i],:])*K(np.linalg.norm(xy_grid[i,:]-xy[idx_neigh_xy_grid[i],:],axis=1,keepdims=True)/bandwidth)
		dxy_px_i[idx_neigh_x_grid[i],:d]  = (1/(N*(bandwidth**(d+2))))*(x_grid[i,:]-x[idx_neigh_x_grid[i],:])*K(np.linalg.norm(x_grid[i,:]-x[idx_neigh_x_grid[i],:],axis=1,keepdims=True)/bandwidth)
		dxy_py_i[idx_neigh_y_grid[i],d:]  = (1/(N*(bandwidth**(d+2))))*(y_grid[i,:]-y[idx_neigh_y_grid[i],:])*K(np.linalg.norm(y_grid[i,:]-y[idx_neigh_y_grid[i],:],axis=1,keepdims=True)/bandwidth)

		Mi += pxy_i*np.log(pxy_i/(px_i*py_i))

		dxy_Mi += dxy_pxy_i*np.log(pxy_i/(px_i*py_i)) + dxy_pxy_i - pxy_i*(dxy_px_i/px_i + dxy_py_i/py_i)


		#histo[np.unravel_index(i,histo.shape)]=pxy_i
	
	Mi*=(h**(2*d))

	dx_Mi=dxy_Mi[:,:d]*(h**(2*d))
	dy_Mi=dxy_Mi[:,d:]*(h**(2*d))

	#plt.scatter(x,y)
	#plt.show()
	#plt.imshow(np.log(histo))
	#plt.show()

	
	
	return Mi,dx_Mi,dy_Mi


def mutual_information_kernel_grid_vectorized(x,y,Ngrid=30):

	N=x.shape[0]

	d=x.shape[1]

	def K(x) : 
		return (1/np.sqrt(2*np.pi))*np.exp(-0.5*(x**2))

	xy=np.concatenate((x,y),axis=1)

	N_out=5

	h=2/(Ngrid-1)
	bandwidth=h

	N_out=int(np.ceil(5*bandwidth/h))
	minval=-1-N_out*h
	maxval=1+N_out*h

	#print(np.linspace(minval,maxval,Ngrid+2*N_out))


	xy_grid=np.reshape(np.array(np.meshgrid(*[np.linspace(minval,maxval,Ngrid+2*N_out) for _ in range(2*d)])),(2*d,(Ngrid+2*N_out)**(2*d))).T
	x_grid=xy_grid[:,:d]
	y_grid=xy_grid[:,d:]

	tree_xy=KDTree(xy)
	tree_x=KDTree(x)
	tree_y=KDTree(y)

	idx_neigh_xy_grid = tree_xy.query_ball_point(xy_grid,5*bandwidth,p=np.inf)
	idx_neigh_x_grid  = tree_x.query_ball_point(x_grid,5*bandwidth,p=np.inf)
	idx_neigh_y_grid  = tree_y.query_ball_point(y_grid,5*bandwidth,p=np.inf)


	# Supposons M = xy_grid.shape[0]
	M = xy_grid.shape[0]

	# Filtrage comme dans ton code
	reduce_idx = np.array([i for i, lst in enumerate(idx_neigh_xy_grid) if lst], dtype=int)

	# Aplatir les listes de voisins pour xy
	grid_idx_xy = np.concatenate([np.full(len(idx_neigh_xy_grid[i]), i, dtype=int)
                              for i in reduce_idx])
	sample_idx_xy = np.concatenate([np.array(idx_neigh_xy_grid[i], dtype=int)
                                for i in reduce_idx])

	# Idem pour x et y
	grid_idx_x = np.concatenate([np.full(len(idx_neigh_x_grid[i]), i, dtype=int)
                             for i in reduce_idx])
	sample_idx_x = np.concatenate([np.array(idx_neigh_x_grid[i], dtype=int)
                               for i in reduce_idx])

	grid_idx_y = np.concatenate([np.full(len(idx_neigh_y_grid[i]), i, dtype=int)
                             for i in reduce_idx])
	sample_idx_y = np.concatenate([np.array(idx_neigh_y_grid[i], dtype=int)
                               for i in reduce_idx])
	
	bw = bandwidth
	const_xy = 1.0 / (N * (bw**(2*d)))
	const_x  = 1.0 / (N * (bw**d))
	const_y  = 1.0 / (N * (bw**d))

	# distances pour xy (toutes paires (i,j) à la fois)
	diff_xy = xy_grid[grid_idx_xy] - xy[sample_idx_xy]          # shape (n_pairs_xy, 2*d)
	r_xy    = np.linalg.norm(diff_xy, axis=1) / bw              # shape (n_pairs_xy,)
	K_xy    = K(r_xy)                                           # shape (n_pairs_xy,)

	pxy_all = np.zeros(M)
	pxy_all += const_xy * np.bincount(grid_idx_xy, weights=K_xy, minlength=M)

	# distances pour x
	diff_x = x_grid[grid_idx_x] - x[sample_idx_x]               # shape (n_pairs_x, d)
	r_x    = np.linalg.norm(diff_x, axis=1) / bw
	K_x    = K(r_x)

	px_all = np.zeros(M)
	px_all += const_x * np.bincount(grid_idx_x, weights=K_x, minlength=M)

	# distances pour y
	diff_y = y_grid[grid_idx_y] - y[sample_idx_y]               # shape (n_pairs_y, d)
	r_y    = np.linalg.norm(diff_y, axis=1) / bw
	K_y    = K(r_y)

	py_all = np.zeros(M)
	py_all += const_y * np.bincount(grid_idx_y, weights=K_y, minlength=M)

	# On récupère pxy_i, px_i, py_i uniquement pour les i utiles
	pxy_i = pxy_all[reduce_idx]
	px_i  = px_all[reduce_idx]
	py_i  = py_all[reduce_idx]


	dxy_pxy = np.zeros((N, 2*d))
	dxy_px  = np.zeros((N, 2*d))
	dxy_py  = np.zeros((N, 2*d))

	# Pour pxy : dérivée par rapport à chaque sample j
	# (je laisse ton facteur en bw^{2d+1}, adapte si tu as corrigé la puissance)
	grad_xy_pairs = (1.0 / (N * (bw**(2*d+1)))) * diff_xy * K_xy[:, None]  # shape (n_pairs_xy, 2*d)

	np.add.at(dxy_pxy, sample_idx_xy, grad_xy_pairs)

	# Pour px : seulement sur les d premières coordonnées
	grad_x_pairs = (1.0 / (N * (bw**(d+1)))) * diff_x * K_x[:, None]       # shape (n_pairs_x, d)
	tmp_x = np.zeros((grad_x_pairs.shape[0], 2*d))
	tmp_x[:, :d] = grad_x_pairs

	np.add.at(dxy_px, sample_idx_x, tmp_x)

	# Pour py : seulement sur les d dernières coordonnées
	grad_y_pairs = (1.0 / (N * (bw**(d+1)))) * diff_y * K_y[:, None]       # shape (n_pairs_y, d)
	tmp_y = np.zeros((grad_y_pairs.shape[0], 2*d))
	tmp_y[:, d:] = grad_y_pairs

	np.add.at(dxy_py, sample_idx_y, tmp_y)


	log_term = np.log(pxy_i / (px_i * py_i))   # shape (len(reduce_idx),)

	Mi = np.sum(pxy_i * log_term) * (h**4)

	dx_Mi=0
	dy_Mi=0
	
	return Mi,dx_Mi,dy_Mi


def entropy_kernel(x,y,idx_sample,K=(lambda x : (1/np.sqrt(2*np.pi))*np.exp(-(0.5*x)**2)),bandwidth='Scott_normalized'):

	N=x.shape[0]
	Nsample=len(idx_sample)

	if(bandwidth=='Scott_normalized'):
		bandwidth=(2*N)**(-1/3)

	xy=np.concatenate((x,y),axis=1)
	xy_sample=xy[idx_sample,:]

	tree_xy=KDTree(xy)
	
	idx_neigh_xy_sample=tree_xy.query_ball_point(xy_sample,5*bandwidth)
	
	E=0

	for i in range(Nsample):
		pxy_i = np.sum((1/(N*(bandwidth**xy.shape[1])))*K(np.linalg.norm(xy_sample[i,:]-xy[idx_neigh_xy_sample[i],:],axis=1)/bandwidth))
		E += np.log(pxy_i)

	E/=Nsample

	dx_E=0*x
	dy_E=0*y
	
	return E,dx_E,dy_E


def mutual_information_tsallis_knn(x,y,k=1,q=1.1):

	xy=np.concatenate((x,y),axis=1)

#	plt.scatter(np.linalg.norm(xy[:,:2],axis=1),np.linalg.norm(xy[:,2:],axis=1))
#	plt.show()
	
	Ent_xy,dxy_Ent_xy=tsallis_entropy_knn(xy,k=k,q=q)
	Ent_x ,dx_Ent_x  =tsallis_entropy_knn(x,k=k,q=q)
	Ent_y ,dy_Ent_y  =tsallis_entropy_knn(y,k=k,q=q)

	dim=x.shape[1]

	Mi=Ent_x+Ent_y-Ent_xy
	dx_Mi=dx_Ent_x-dxy_Ent_xy[:,:dim]
	dy_Mi=dy_Ent_y-dxy_Ent_xy[:,dim:]

	return Mi,dx_Mi,dy_Mi

def joint_entropy_kozachenko_leonenko(x,y,k=1,eps=0):

	xy=np.concatenate((x,y),axis=1)
	
	Ent_xy,dxy_Ent_xy=entropy_kozachenko_leonenko(xy,k=k,eps=eps)
	
	dim=x.shape[1]

	Mi=Ent_xy
	dx_Mi=dxy_Ent_xy[:,:dim]
	dy_Mi=dxy_Ent_xy[:,dim:]

	return Mi,dx_Mi,dy_Mi

def joint_entropy_Tsallis_knn(x,y,k=1,q=1.1):

	xy=np.concatenate((x,y),axis=1)
	
	Ent_xy,dxy_Ent_xy=tsallis_entropy_knn(xy,k=k,q=q)

	dim=x.shape[1]

	E=Ent_xy
	dx_E=dxy_Ent_xy[:,:dim]
	dy_E=dxy_Ent_xy[:,dim:]

	return E,dx_E,dy_E

def modulus_correlation(x,y):
	pass


# Full Chat-GPT
def inverse_map(N):
    N = np.asarray(N)
    n = len(N)

    # indices triés selon N[j]
    order = np.argsort(N)
    sorted_vals = N[order]

    # limites où la valeur change
    bounds = np.flatnonzero(np.diff(sorted_vals)) + 1

    # regrouper les indices
    groups = np.split(order, bounds)

    # Maintenant, il faut que groups[i] existe pour chaque i de 0 à n-1
    # même s'il est vide.
    result = [np.array([], dtype=int) for _ in range(n)]
    uniq_vals = np.unique(N)

    for val, grp in zip(uniq_vals, groups):
        result[val] = grp

    return result

# Full Chat-GPT
def Vd(d):
    return (np.pi**(d/2)) / (gamma(1 + d/2)*(2**d))

def entropy_kozachenko_leonenko(x,k=1,eps=0):
	tree=KDTree(x)
	d,i=tree.query(x,k=k+1)

	dists=d[:,-1]+eps
	idx=i[:,-1]
	x_nei=x[idx,:]

	idx_inv=inverse_map(idx)

	Ent=(x.shape[1]/x.shape[0])*np.sum(np.log(dists))+digamma(x.shape[0])-digamma(k)+np.log(Vd(x.shape[1]))

	dxEnt=0*x

	for i in range(x.shape[0]):
		dxEnt[i]=(x[i,:]-x_nei[i,:])/((np.linalg.norm(x[i,:]-x_nei[i,:])+eps)**2)
		for j in idx_inv[i]:
			dxEnt[i]+=(x[i,:]-x[j,:])/((np.linalg.norm(x[i,:]-x[j,:])+eps)**2)

	dxEnt=dxEnt*(x.shape[1]/x.shape[0])

	return Ent,dxEnt

# From https://infomeasure.readthedocs.io/en/0.5.0/guide/entropy/tsallis/

def tsallis_entropy_knn(x,k=1,q=1.1):
	tree=KDTree(x)
	d,i=tree.query(x,k=k+1)

	dists=d[:,-1]

	idx=i[:,-1]

	m=x.shape[1]
	N=x.shape[0]

	V_m=Vd(m)
	C_k=(gamma(k)/gamma(k+1-q))**(1/(1-q))
	
	x_nei=x[idx,:]
	idx_inv=inverse_map(idx)

	zeta=(N-1)*C_k*V_m*(dists**m)

	Ihat=(1/N)*np.sum(zeta**(1-q))
	Ent=(1-Ihat)/(q-1)

	dxEnt=0*x

	power=m*(1-q)

	for i in range(x.shape[0]):
		dxEnt[i]=((x[i,:]-x_nei[i,:])/(np.linalg.norm(x[i,:]-x_nei[i,:])**(1-(power-1))))*power
		for j in idx_inv[i]:
			dxEnt[i]+=((x[i,:]-x[j,:])/(np.linalg.norm(x[i,:]-x[j,:])**(1-(power-1))))*power

	dxEnt=-dxEnt*(1/N)*(((N-1)*C_k*V_m)**(1-q))/(q-1)

	return Ent,dxEnt


def test_cork(f):

	def gaussian_cork(x):
		return np.exp(-(x**2)+1j*5*x)


	X=np.sort(8*np.random.rand(200)-4)
	Y=gaussian_cork(X)

	shift=np.linspace(-5,5,1000)

	err=[]
	for s in shift:
		print(s)
		Ys=gaussian_cork(X+s)
		MI=f(np.array([np.real(Y),np.imag(Y)]).T,np.array([np.real(Ys),np.imag(Ys)]).T)
		err.append(MI[0])

	plt.plot(shift,err,label=f.__name__)


def test_cork2():

	def gaussian_cork(x):
		return np.exp(-(x**2))


	X=np.sort(8*np.random.rand(100)-4)
	Y=gaussian_cork(X)

	shift=np.linspace(-5,5,1000)

	err=[]
	for s in shift:
		print(s)
		Ys=gaussian_cork(X+s)

		MI=mutual_information_kernel_grid(np.array([Y]).T,np.array([Ys]).T,Ngrid=10)

		err.append(MI[0])

	plt.plot(shift,err)



def test_conv():

	x0=np.random.rand(1000)
	y=np.random.rand(1000)

	def f(x):
		X=np.reshape(x,(1000,1))
		Y=np.reshape(y,(1000,1))
		Mi,dx_Mi,dy_Mi=mutual_information_kernel_vectorized(X,Y)
		return Mi

	def fp(x):
		X=np.reshape(x,(1000,1))
		Y=np.reshape(y,(1000,1))
		Mi,dx_Mi,dy_Mi=mutual_information_kernel_vectorized(X,Y)
		return dx_Mi.flatten()

	epsilons=[10**(-i) for i in range(1,12)]
	err_grad=[]
	for eps in epsilons:
		print(eps)
		err_grad.append(check_grad(f,fp,x0,epsilon=eps,direction='random'))

	plt.loglog(epsilons,err_grad,'.-')
	plt.show()
	

if __name__ == '__main__':
	for f in [mutual_information_kraskov_stoegbauer_grassberger,mutual_information_kernel_vectorized]:
		test_cork(f)

	plt.legend()
	plt.show()

