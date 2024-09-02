import numpy as np
from help_fcts import r2, kgd, kpr, kmrh, kmrt, kqrz, kqrt, krr, cv10, make_data_real
import time
import pickle
import sys

sigmas=np.geomspace(0.01, 1000, 30)
lbdas=np.geomspace(1e-6, 10, 30)

taus=[0.25, 0.5, 0.75]
k_hubs=1.345*np.array([0.5,1,2])
k_tuks=4.685*np.array([0.5,1,2])

kr_names=['ksgd', 'kpr', 'kmrh', 'kmrt', 'kqrz', 'kqrt', 'kgd', 'krr']

REAL_DATA='real_data'
data='house'
seed=1
nu=100
n_samps=100
cauchy=0.1

for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

kr_dict={}
for kr_name in kr_names:
  kr_dict[kr_name]={}

if 'ksgd' in kr_names:
  kr_dict['ksgd']['fun']=kgd
  kr_dict['ksgd']['lbdas']=[None]
  kr_dict['ksgd']['hp3s']=['sgd']

if 'kpr' in kr_names:
  kr_dict['kpr']['fun']=kpr
  kr_dict['kpr']['lbdas']=lbdas
  kr_dict['kpr']['hp3s']=['linf']

if 'kmrh' in kr_names:
  kr_dict['kmrh']['fun']=kmrh
  kr_dict['kmrh']['lbdas']=lbdas
  kr_dict['kmrh']['hp3s']=k_hubs

if 'kmrt' in kr_names:
  kr_dict['kmrt']['fun']=kmrt
  kr_dict['kmrt']['lbdas']=lbdas
  kr_dict['kmrt']['hp3s']=k_tuks

if 'krr' in kr_names:
  kr_dict['krr']['fun']=krr
  kr_dict['krr']['lbdas']=lbdas
  kr_dict['krr']['hp3s']=['None']

if 'kgd' in kr_names:
  kr_dict['kgd']['fun']=kgd
  kr_dict['kgd']['lbdas']=[None]
  kr_dict['kgd']['hp3s']=['gd']

if 'kqrz' in kr_names:
  kr_dict['kqrz']['fun']=kqrz
  kr_dict['kqrz']['lbdas']=lbdas
  kr_dict['kqrz']['hp3s']=taus

if 'kqrt' in kr_names:
  kr_dict['kqrt']['fun']=kqrt
  kr_dict['kqrt']['lbdas']=lbdas
  kr_dict['kqrt']['hp3s']=taus


data_dict={}
for kr_name in kr_names:
  data_dict[kr_name]={}

X_tr, y_tr, X_te, y_te = make_data_real(data, n_samps, seed, cauchy)

for kr_name in kr_names:
  t1=time.time()
  sigma_opt, lbda_iter_opt, hp3_opt = cv10(X_tr, y_tr, sigmas, kr_dict[kr_name]['lbdas'], kr_dict[kr_name]['hp3s'], seed, kr_dict[kr_name]['fun'], nu)
  fh = kr_dict[kr_name]['fun'](X_te, X_tr, y_tr, sigma_opt, lbda_iter_opt, hp3_opt, nu=nu)[0]
  data_dict[kr_name]['time']=time.time()-t1
  data_dict[kr_name]['r2']=r2(y_te, fh)
  data_dict[kr_name]['sigma']=sigma_opt
  data_dict[kr_name]['lbda']=lbda_iter_opt
  data_dict[kr_name]['hp3']=hp3_opt
  print(f'Data: {data:<7}. Alg: {kr_name:<4}. Time: {time.time()-t1:<4.3g}. R2: {r2(y_te,fh):<+7.3g}. Sig: {sigma_opt:<4.3g}. Lbda: {lbda_iter_opt:<4.3g}. HP3: {hp3_opt:<4}')

fi=open(REAL_DATA+'/'+data+'-'+str(cauchy)+'_'+str(nu)+'_'+str(seed)+'.pkl','wb')
pickle.dump(data_dict,fi)
fi.close()
