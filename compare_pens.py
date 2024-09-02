import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys
from help_fcts import kgd, kpr, kpr_f, krr, kgf

def f(x):
  y=np.sin(np.pi*x)
  return y

seed=0
np.random.seed(seed)
x_tr=np.array([0.1,0.5,1.0,1.5,2.1,2.5,2.9]).reshape((-1,1))
n=x_tr.shape[0]
y_tr=f(x_tr)+np.random.normal(0,.01,x_tr.shape)
y_tr[1]+=3
x_te=np.linspace(0,3,1001).reshape((-1,1))
sigma=0.3
fh_0, alphah_0=krr(x_te,x_tr,y_tr,sigma,0)



labs1=['Observed Data', 'Non- and Fully Reguralized Solutions','Gradient-Based Optimization', 'Explicit Regularization']
lines1=[plt.plot(0,0,'ok')[0]]
plt.cla()
for c in ['C7','C2','C1']:
  lines1.append(Line2D([0],[0],color=c,lw=2))

labs2=['Non-Reguralized Solution','Gradient-Based Optimization', 'Explicit Regularization']
lines2=[]
for c in ['C7','C2','C1']:
  lines2.append(Line2D([0],[0],color=c,lw=4))

labs3=['Observed Data', 'Non- and Fully Reguralized Solutions', 'Explicit Regularization']
lines3=[plt.plot(0,0,'ok')[0]]
plt.cla()
for c in ['C7','C2']:
  lines3.append(Line2D([0],[0],color=c,lw=2))


fig1,axss1=plt.subplots(2,3,figsize=(10,5))
fig2,axss2=plt.subplots(2,3,figsize=(10,5))
fig3,axss3=plt.subplots(2,3,figsize=(10,5))

for axss in [axss1,axss2]:
  axss[0,0].set_title('KGF and KRR',fontsize=13)
  axss[0,1].set_title('KSGD and K$\\ell_\\infty$R',fontsize=13)
  axss[0,2].set_title('KCD and K$\\ell_1$R',fontsize=13)

axss3[0,0].set_title('KRR',fontsize=13)
axss3[0,1].set_title('K$\\ell_\\infty$R',fontsize=13)
axss3[0,2].set_title('K$\\ell_1$R',fontsize=13)

axss1[0,0].set_ylabel('$\\hat{f}$',fontsize=13)
axss1[1,0].set_ylabel('$\\hat{f}$',fontsize=13)
axss2[0,0].set_ylabel('$\\hat{\\alpha}$',fontsize=13)
axss2[1,0].set_ylabel('$\\hat{\\alpha}$',fontsize=13)
axss3[0,0].set_ylabel('$\\hat{f}$',fontsize=13)
axss3[1,0].set_ylabel('$\\hat{f}$',fontsize=13)


BW=0.35
LW=2
fig1.legend(lines1, labs1, loc='lower center', ncol=3)
fig1.tight_layout()
fig1.subplots_adjust(bottom=0.17)
fig2.legend(lines2, labs2, loc='lower center', ncol=3)
fig2.tight_layout()
fig2.subplots_adjust(bottom=0.13)
fig3.legend(lines3, labs3, loc='lower center', ncol=3)
fig3.tight_layout()
fig3.subplots_adjust(bottom=0.17)

lbdass_a=[[1,0.04,0.02],[.4,0.01,0.007]]
lbdass_f=[[.01,6,0.02],[.003,.7,0.01]]

for axs1,axs2,axs3,lbdas_a,lbdas_f in zip(axss1,axss2,axss3,lbdass_a,lbdass_f):
  for ax1, ax2, ax3, nrm, alg,lbda_a,lbda_f in zip(axs1,axs2,axs3,['l2','linf','l1'], ['gd','sgd','cd'],lbdas_a,lbdas_f):
    if nrm=='l2':
      fh_l, alphah_l=krr(x_te,x_tr,y_tr,sigma,lbda_a)
      fh_t, alphah_t=kgf(x_te,x_tr,y_tr,sigma,1/lbda_a)
    else:
      fh_l, alphah_l= kpr(x_te,x_tr,y_tr,sigma,lbda_a,nrm)
      _, _, fh_t, alphah_t= kgd(x_te,x_tr,y_tr,sigma,None,alg,fh_l,auto=True, max_c=1)

    fh_l_f, _ = kpr_f(x_te,x_tr,y_tr,sigma,lbda_f,nrm)
    
    ax1.plot(x_te,fh_0,'C7', lw=4)
    ax1.plot(x_te,np.zeros(x_te.shape),'C7', lw=4)
    ax1.plot(x_tr,y_tr,'ok')
    ax1.plot(x_te,fh_l,'C1',lw=3.5)
    ax1.plot(x_te,fh_t,'C2',lw=2)
    fig1.savefig('figures/compare_pen_af.pdf')
    
    ax2.bar(1+np.arange(n),np.squeeze(alphah_0),color='C7', width=2*BW)
    ax2.bar(1+np.arange(n)-BW/2,np.squeeze(alphah_t),color='C2', width=BW)
    ax2.bar(1+np.arange(n)+BW/2,np.squeeze(alphah_l),color='C1', width=BW)
    ax2.set_xticks(1+np.arange(n))
    fig2.savefig('figures/compare_pen_a.pdf')
    
    ax3.plot(x_te,fh_0,'C7', lw=4)
    ax3.plot(x_te,np.zeros(x_te.shape),'C7', lw=4)
    ax3.plot(x_tr,y_tr,'ok')
    ax3.plot(x_te,fh_l_f[n:],'C2',lw=3.5)
    fig3.savefig('figures/compare_pen_f.pdf')

