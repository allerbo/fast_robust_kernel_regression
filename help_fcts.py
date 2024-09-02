import numpy as np
import pandas as pd
from proximal import prox_grad
from sklearn import datasets
import sys, subprocess

from rpy2.robjects import r, numpy2ri
from rpy2.robjects.packages import importr
from rpy2.rinterface_lib import callbacks
callbacks.consolewrite_warnerror = lambda *args: None
numpy2ri.activate()
importr('fastkqr')

def r2(y,y_hat):
  if len(y.shape)==1:
    y=y.reshape((-1,1))
  if len(y_hat.shape)==1:
    y_hat=y_hat.reshape((-1,1))
  return 1-np.mean((y-y_hat)**2)/np.mean((y-np.mean(y))**2)

def mse(y,y_hat):
  return np.mean(np.square(y-y_hat))

def kgd(x_te, x_tr, y_tr, sigma, n_iters=None, alg='sgd', y_te=None, auto=False, step_size=0.01, t_max=100, nu=100, max_c=100):
  K_tr=kern(x_tr, x_tr, sigma, nu)
  K_te=kern(x_te, x_tr, sigma, nu)
  
  alphah=np.zeros(y_tr.shape)
  alphah_old=np.zeros(y_tr.shape)
  if auto:
    best_mse=np.inf
    mse_counter=0
    n_iters=int(t_max/step_size)
  for n_iter in range(n_iters):
    grad=K_tr@alphah-y_tr
    if alg=='sgd':
      alphah-=step_size*np.sign(grad)
    elif alg=='gd':
      alphah-=step_size*grad
    elif alg=='cd':
      alphah-=step_size*np.sign(grad)*(np.abs(grad)==np.max(np.abs(grad)))
    if auto:
      mse_te=mse(y_te,K_te@alphah)
      if mse_te<best_mse:
        best_mse=mse_te
        best_n_iter=n_iter
        best_fh=K_te@alphah
        best_alphah=alphah
        mse_counter=0
      mse_counter+=1
      if mse_counter>max_c:
        break
  if auto:
    return best_mse, best_n_iter, best_fh, best_alphah
  return K_te@alphah, alphah

def kpr_f(x_te,x_tr,y_tr,sigma,lbda,nrm='linf',y_te=None, auto=False, step_size=0.01, t_max=1000, nu=100):
  K_tr=kern(x_tr,x_tr,sigma,nu)
  K_te=kern(x_te,x_tr,sigma,nu)
  
  prox_obj=prox_grad(np.vstack((K_tr,K_te)),y_tr,lbda,nrm,'pred',step_size)
  fh_old=np.ones(K_te.shape[0])
  for ii in range(t_max):
    prox_obj.prox_step()
    fh=prox_obj.get_fh()
    if np.linalg.norm(fh-fh_old)<1e-5:
      break
    fh_old=np.copy(fh)
  if auto:
    return mse(y_te, fh_te), ii, None, None
  return fh, None
  
def kpr(x_te,x_tr,y_tr,sigma,lbda,nrm='linf',y_te=None, auto=False, step_size=0.01, t_max=1000, nu=100):
  K_tr=kern(x_tr,x_tr,sigma,nu)
  K_te=kern(x_te,x_tr,sigma,nu)
  
  prox_obj=prox_grad(K_tr,y_tr,lbda,nrm,'par',step_size)
  alphah_old=np.ones(K_tr.shape[0])
  for ii in range(t_max):
    prox_obj.prox_step()
    alphah=prox_obj.get_alphah()
    #if np.allclose(alphah_old, alphah, rtol=0.0001, atol=0.0001):
    if np.linalg.norm(alphah-alphah_old)<1e-5:
      break
    alphah_old=np.copy(alphah)
  fh_te=K_te@alphah
  if auto:
    return mse(y_te, fh_te), ii, None, None
  return fh_te, alphah

def kmrh(x_te,x_tr,y_tr,sigma,lbda,k_hub,y_te=None, auto=False, t_max=100, nu=100):
  return kmr(x_te,x_tr,y_tr,sigma,lbda,k_hub,True,y_te, auto, t_max, nu)

def kmrt(x_te,x_tr,y_tr,sigma,lbda,k_tuk,y_te=None, auto=False, t_max=100, nu=100):
  return kmr(x_te,x_tr,y_tr,sigma,lbda,k_tuk,False,y_te, auto, t_max, nu)

def kmr(x_te,x_tr,y_tr,sigma,lbda,hp3,hub,y_te=None, auto=False, t_max=100, nu=100):
  K_tr=kern(x_tr, x_tr, sigma, nu)
  K_te=kern(x_te, x_tr, sigma, nu)

  alphah=np.zeros((K_tr.shape[0],1))
  alphah_old=np.ones((K_tr.shape[0],1))
  
  for ii in range(t_max):
    res = y_tr-K_tr@alphah
    k_st=hp3*np.std(res)
    if hub:
      W=np.diag(np.minimum(1, np.squeeze(k_st/np.abs(res))))
    else:
      W=np.diag(np.maximum(0, np.squeeze(1-(res/k_st)**2)))
    alphah=W@np.linalg.solve(W@K_tr@W+lbda*np.eye(K_tr.shape[0]),W@y_tr)
    if np.linalg.norm(alphah-alphah_old)<1e-5:
      break
    alphah_old=np.copy(alphah)
  fh_te=K_te@alphah
  if auto:
    return mse(y_te, fh_te), ii, None, None
  return fh_te, alphah

def kqrz(x_te,x_tr,y_tr,sigma,lbda,tau, y_te=None, auto=False, t_max=100, nu=100):
  K_tr=kern(x_tr, x_tr, sigma, nu)
  K_te=kern(x_te, x_tr, sigma, nu)
  
  alphah=np.zeros((K_tr.shape[0],1))
  alphah_old=np.ones((K_tr.shape[0],1))
  for ii in range(t_max):
    abs_res=np.abs(y_tr-K_tr@alphah)
    U=np.diag(1/(4*(1e-5+np.squeeze(abs_res))))
    v=0.5*(y_tr/(1e-5+abs_res)+2*tau-1)
    alphah=np.linalg.solve(2*U@K_tr+lbda*np.eye(K_tr.shape[0]),v)
    if np.linalg.norm(alphah-alphah_old)<1e-5:
      break
    alphah_old=np.copy(alphah)
  fh_te=K_te@alphah
  if auto:
    return mse(y_te, fh_te), ii, None, None
  return fh_te, alphah

def kqrt(x_te,x_tr,y_tr,sigma,lbda,tau, y_te=None, auto=False, t_max=100, nu=100):
  K_te=kern(x_te, x_tr, sigma, nu)
  
  bh_alphah=r.coef(r.kqr(x_tr,y_tr,lbda,tau,sigma=sigma))
  bh=bh_alphah[0]
  alphah=bh_alphah[1:]
  
  fh_te=K_te@alphah+bh
  if auto:
    return mse(y_te, fh_te), 0, None, None
  return fh_te, alphah


def krr(x_te,x_tr,y_tr,sigma,lbda,hp3=None, y_te=None, auto=False, nu=100):
  K_tr=kern(x_tr,x_tr,sigma, nu)
  K_te=kern(x_te,x_tr,sigma, nu)
  alphah=np.linalg.solve(K_tr+lbda*np.eye(K_tr.shape[0]),y_tr)
  fh_te=K_te@alphah
  if auto:
    return mse(y_te, fh_te), 0, None, None
  return fh_te, alphah

def kgf(x_te,x_tr,y_tr,sigma,t,hp3=None, y_te=None, auto=False, nu=100):
  from scipy.linalg import expm
  K_tr=kern(x_tr,x_tr,sigma, nu)
  K_te=kern(x_te,x_tr,sigma, nu)
  alphah=np.linalg.inv(K_tr)@(np.eye(K_tr.shape[0])-expm(-t*K_tr))@y_tr
  fh_te=K_te@alphah
  if auto:
    return mse(y_te, fh_te), 0, None, None
  return fh_te, alphah

def cv10(x,y, sigmas, lbdas, hp3s, seed, kr_fun, nu):
  n=x.shape[0]
  np.random.seed(seed)
  per=np.random.permutation(n)
  folds=np.array_split(per,10)
  best_params=(np.inf,None,None,None)
  iii=0
  for sigma in sigmas:
    iii+=1
    sys.stdout.write(str(iii)+' '+str(np.round(sigma,3))+' '+str(np.round(best_params[0],3))+'\r')
    sys.stdout.flush()
    for lbda in lbdas:
      for hp3 in hp3s:
        mses=[]
        n_iters=[]
        for v_fold in range(len(folds)):
          t_folds=np.concatenate([folds[t_fold] for t_fold in range(len(folds)) if v_fold != t_fold])
          v_folds=folds[v_fold]
          x_tr=x[t_folds,:]
          y_tr=y[t_folds,:]
          x_val=x[v_folds,:]
          y_val=y[v_folds,:]
          mse_val, n_iter, _, _=kr_fun(x_val, x_tr, y_tr, sigma, lbda, hp3, y_val, auto=True, nu=nu)
          mses.append(mse_val)
          n_iters.append(n_iter)
        mean_mse=np.mean(mses)
        mean_iters=int(np.mean(n_iters))
        if mean_mse<best_params[0]:
          if lbda is None:
            best_params=(mean_mse, sigma, mean_iters, hp3)
          else:
            best_params=(mean_mse, sigma, lbda, hp3)
  return best_params[1], best_params[2], best_params[3]

def make_data_real(data, n_samps=100, seed=None, cauchy=0):
  FRAC=0.8
  if data=='super':
    dm_all=pd.read_csv('csv_files/super.csv',sep=',').to_numpy()
    dm_all=np.roll(dm_all,1,1)
  elif data=='temp':
    dm_all=pd.read_csv('csv_files/uktemp.csv',sep=',').to_numpy()
  elif data=='house':
    house=datasets.fetch_california_housing()
    dm_all=np.hstack((house.target.reshape((-1,1)),house.data))
  elif data=='airfoil':
    dm_all=pd.read_csv('csv_files/airfoil.csv',sep=',').to_numpy()
    dm_all=np.roll(dm_all,1,1)
  elif data=='steel':
    dm_all=pd.read_csv('csv_files/steel.csv',sep=',').to_numpy()
  
  if not seed is None:
    np.random.seed(seed)
  
  np.random.shuffle(dm_all)
  dm=dm_all[:n_samps,:]
  
  X=dm[:,1:]
  X=(X-np.mean(X, 0))/np.std(X,0)
  n=X.shape[0]
  X_tr=X[:int(FRAC*n),:]
  X_te=X[int(FRAC*n):,:]
  
  y=dm[:,0].reshape((-1,1))
  y=y-np.mean(y)
  y*=(1+cauchy*np.abs(np.random.standard_cauchy(y.shape)))
  
  y_tr=y[:int(FRAC*n),:]
  y_te=y[int(FRAC*n):,:]
  
  return X_tr, y_tr, X_te, y_te


def make_data_synth(data, seed=None):
  if not seed is None:
    np.random.seed(seed)
  N_TR=100
  N_TE=1000
  X_MAX=10
  if data=='sin':
    def fy(x):
      return np.sin(1/4*2*np.pi*x)
  elif data=='exp':
    def fy(x):
      return np.exp(-5*x**2)
  x_tr=np.random.uniform(-X_MAX,X_MAX,N_TR).reshape((-1,1))
  if data=='sin':
    y_tr=fy(x_tr)+0.1*np.random.standard_cauchy((N_TR,1))
    sigma_bounds=[0.1,10]
  else:
    x_tr[0]=0
    y_tr=fy(x_tr)+np.random.normal(0,0.1,(N_TR,1))
    sigma_bounds=[0.1,1]
  x_te=np.linspace(-X_MAX, X_MAX, N_TE).reshape((-1,1))
  y_te=fy(x_te)
  lbda_bounds=[1e-5,10]
  return x_tr, y_tr, x_te, y_te, lbda_bounds, sigma_bounds



def kern(X,Y,sigma, nu=np.inf):
  X2=np.sum(X**2,1).reshape((-1,1))
  XY=X.dot(Y.T)
  Y2=np.sum(Y**2,1).reshape((-1,1))
  D2=X2-2*XY+Y2.T
  D=np.sqrt(D2+1e-10)
  if nu==0.5:      #Laplace
    return np.exp(-D/sigma)
  elif nu==1.5:
    return (1+np.sqrt(3)*D/sigma)*np.exp(-np.sqrt(3)*D/sigma)
  elif nu==2.5:
    return (1+np.sqrt(5)*D/sigma+5*D2/(3*sigma**2))*np.exp(-np.sqrt(5)*D/sigma)
  elif nu==10:     #Cauchy (could have been any number, but I chose 10 for no particular reason)
    return 1/(1+D2/sigma**2)
  else:            #Gaussian
    return np.exp(-0.5*D2/sigma**2)

