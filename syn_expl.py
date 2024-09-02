import numpy as np
from help_fcts import r2, kgd, kpr, krr, cv10, make_data_synth
import sys
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D



lines=[Line2D([0],[0],color='C7',lw=6),plt.plot(0,0,'ok')[0]]
plt.cla()
for ls,c in zip(['-','--','--',':'],['C2','C9','C1','C3']):
  lines.append(Line2D([0],[0],color=c,ls=ls,lw=2))

labs=['True Function', 'Observed Data', 'KSGD/KCD', 'K$\\ell_\\infty$R/K$\\ell_1$R', 'KGD', 'KRR']

#47
#35


sigmas=np.geomspace(0.01, 1000, 30)
lbdas=np.geomspace(1e-6, 10, 30)

kr_names=['kxgd', 'kpr', 'kgd', 'krr']


for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

kr_dict={}
for kr_name in kr_names:
  kr_dict[kr_name]={}

if 'kxgd' in kr_names:
  kr_dict['kxgd']['fun']=kgd
  kr_dict['kxgd']['lbdas']=[None]
  kr_dict['kxgd']['hp3s']=['sgd']

if 'kpr' in kr_names:
  kr_dict['kpr']['fun']=kpr
  kr_dict['kpr']['lbdas']=lbdas
  kr_dict['kpr']['hp3s']=['linf']

if 'krr' in kr_names:
  kr_dict['krr']['fun']=krr
  kr_dict['krr']['lbdas']=lbdas
  kr_dict['krr']['hp3s']=[None]

if 'kgd' in kr_names:
  kr_dict['kgd']['fun']=kgd
  kr_dict['kgd']['lbdas']=[None]
  kr_dict['kgd']['hp3s']=['gd']


nu=100

fig,axs=plt.subplots(2,1,figsize=(8,5))
fhs=[]
for ax,alg, nrm,data,seed, title in zip(axs,['sgd','cd'],['linf','l1'], ['sin','exp'],[47,35], ['KSGD and K$\\ell_\\infty$R','KCD and K$\\ell_1$R']):
  X_tr, y_tr, X_te, y_te, lbda_bounds, sigma_bounds = make_data_synth(data,seed)
  kr_dict['kxgd']['hp3s']=[alg]
  kr_dict['kpr']['hp3s']=[nrm]
  ax.cla()
  ax.plot(X_te,y_te,'C7',lw=6)
  for kr_name, col,lw in zip(kr_names, ['C2-','C9--','C1--','C3:'],[4,3,2,2]):
    sigma_opt, lbda_iter_opt, hp3_opt = cv10(X_tr, y_tr, sigmas, kr_dict[kr_name]['lbdas'], kr_dict[kr_name]['hp3s'], seed, kr_dict[kr_name]['fun'], nu)
    fh = kr_dict[kr_name]['fun'](X_te, X_tr, y_tr, sigma_opt, lbda_iter_opt, hp3_opt, nu=nu)[0]
    #fhs.append(fh)
    ax.plot(X_te,fh,col,lw=lw)
  
  ax.plot(X_tr,y_tr,'xk')
  ax.set_title(title)
    
  fig.legend(lines, labs, loc='lower center', ncol=len(labs))
  fig.tight_layout()
  fig.subplots_adjust(bottom=0.12)
  fig.savefig('figures/syn_expl.pdf')

#fig,axs=plt.subplots(2,1,figsize=(8,5))
#ii=0
#for ax,alg, nrm,data,seed, title in zip(axs,['sgd','cd'],['linf','l1'], ['sin','exp'],[47,35], ['K$\\ell_\\infty$R and KSGD','K$\\ell_1$R and KCD']):
#  X_tr, y_tr, X_te, y_te, lbda_bounds, sigma_bounds = make_data_synth(data,seed)
#  ax.cla()
#  ax.plot(X_te,y_te,'C7',lw=6)
#  for kr_name, col,lw in zip(kr_names, ['C2-','C9--','C1--','C3:'],[4,3,2,2]):
#    ax.plot(X_te,fhs[ii],col,lw=lw)
#    ii+=1
#  
#  ax.plot(X_tr,y_tr,'xk')
#  ax.set_title(title)
#    
#  fig.legend(lines, labs, loc='lower center', ncol=len(labs))
#  fig.tight_layout()
#  fig.subplots_adjust(bottom=0.12)
#  fig.savefig('figures/syn_expl.pdf')

