import numpy as np
import pickle
from scipy.stats import wilcoxon
from glob import glob
import sys
import re


REAL_DATA='real_data'

Q1=0.025


for arg in range(1,len(sys.argv)):
  exec(sys.argv[arg])

for nu_str in ['_0.5_', '_1.5_', '_2.5_', '_10_', '_100_']:
  data_sets=sorted(list(set(map(lambda s: re.split('/|_',s)[2],glob(REAL_DATA+'/*')))))
  
  data_dict={}
  for data in data_sets:
    data_dict[data]={}
    seeds=list(map(lambda s: s.split('_')[-1],glob(REAL_DATA+'/'+data+'_*')))
    
    for seed in seeds:
      fi=open(REAL_DATA+'/'+data+nu_str+seed,'rb')
      data_dict_seed=pickle.load(fi)
      fi.close()
      for kr_name in data_dict_seed.keys():
        if not kr_name in data_dict[data].keys():
          data_dict[data][kr_name]={}
        for metric_name in data_dict_seed[kr_name].keys():
          if not metric_name in data_dict[data][kr_name].keys():
            data_dict[data][kr_name][metric_name]=[]
          data_dict[data][kr_name][metric_name].append(data_dict_seed[kr_name][metric_name])
  
  tab0=[]
  tab1=[]
  
  fun_titles={'ksgd': 'KSGD', 'kpr': 'K$\\ell_\\infty$R', 'kmrh': 'KMR-H', 'kmrt': 'KMR-T', 'kqrz': 'KQR-A', 'kqrt': 'KQR-B', 'kgd': 'KGD', 'krr': 'KRR'}
  data_titles={'airfoil': '\\makecell{Airfoil Sound\\\\Pressure}', 'house': '\\makecell{California\\\\House Values}', 'temp': '\\makecell{U.K.\\\\Temperature}', 'steel': '\\makecell{Steel Energy\\\\Consumption}', 'super': '\\makecell{Superconductor\\\\Critical\\\\Temperature}'}
  
  seen_data=[]
  for data in data_sets:
    for kr_fun in ['ksgd', 'kpr', 'kmrh', 'kmrt', 'kqrz', 'kqrt', 'kgd', 'krr']:
      if kr_fun=='kqrt' and not nu_str=='_100_':
        continue
      q1_r2=np.nanquantile(data_dict[data][kr_fun]['r2'],Q1)
      q2_r2=np.nanquantile(data_dict[data][kr_fun]['r2'],0.5)
      q3_r2=np.nanquantile(data_dict[data][kr_fun]['r2'],1-Q1)
      q1_time=np.nanquantile(data_dict[data][kr_fun]['time'],Q1)
      q2_time=np.nanquantile(data_dict[data][kr_fun]['time'],0.5)
      q3_time=np.nanquantile(data_dict[data][kr_fun]['time'],1-Q1)
      q1_sig=np.nanquantile(data_dict[data][kr_fun]['sigma'],Q1)
      q2_sig=np.nanquantile(data_dict[data][kr_fun]['sigma'],0.5)
      q3_sig=np.nanquantile(data_dict[data][kr_fun]['sigma'],1-Q1)
      q1_lab=np.nanquantile(data_dict[data][kr_fun]['lbda'],Q1)
      q2_lab=np.nanquantile(data_dict[data][kr_fun]['lbda'],0.5)
      q3_lab=np.nanquantile(data_dict[data][kr_fun]['lbda'],1-Q1)
      
      if not data in seen_data:
        seen_data.append(data)
        if nu_str=='_100_':
          data_str='\\multirow{9}{*}{'+data_titles[data.split('-')[0]]+'}\n'
        else:
          data_str='\\multirow{8}{*}{'+data_titles[data.split('-')[0]]+'}\n'
      else:
        data_str=''
      if kr_fun in ['kmrh', 'kgd']:
        data_str+='\\cline{2-4}\n'
      data_str+=f'& {fun_titles[kr_fun]:<15} & ${q2_time:#.3g},\\ ({q1_time:#.3g}, {q3_time:#.3g})$ & ${q2_r2:<5.2f},\\ ({q1_r2:<7.2f}, {q3_r2:<5.2f})$ \\\\'.replace('.,',',').replace('.)',')')
      if kr_fun in ['krr']:
        data_str+='\n\\hline'
      if data[-1:]=='0':
        tab0.append(data_str)
      elif data[-1:]=='1':
        tab1.append(data_str)
  
  for tab, noise in zip([tab0, tab1],['0','1']):
    if noise=='1':
      noise_str='with'
      last_sent=' The two non-robust methods, KGD/KRR, do not perform as well as the robust methods.'
    elif noise=='0':
      noise_str='without'
      last_sent=''
    if nu_str=='_10_':
      kernel_str=', for the Cauchy kernel'
      n_mods='five'
    elif nu_str=='_0.5_':
      kernel_str=', for $\\nu=0.5$ (Laplace kernel)'
      n_mods='five'
    elif nu_str=='_1.5_':
      kernel_str=', for $\\nu=1.5$'
      n_mods='five'
    elif nu_str=='_2.5_':
      kernel_str=', for $\\nu=2.5$'
      n_mods='five'
    else:
      kernel_str=''
      n_mods='six'
    print('\\begin{table}')
    print('\\caption{The 2.5th, 50th and 97.5th percentiles of computation time and test $R^2$ for the different methods and data sets, \\textbf{'+noise_str+'} amplified outliers' + kernel_str +'. The '+ n_mods +' robust methods perform very similarly in terms of test $R^2$, while KSGD performs one to two orders of magnitude faster.'+last_sent+'}')

    print('\\center')
    print('\\begin{tabular}{l|l|l|l}')
    print('\\hline')
    print('Data & Method & \\makecell{Computation Time [s]\\\\50\\%,\\ (2.5\\%,\\ 97.5\\%)} & \\makecell{Test $R^2$\\\\50\\%,\\ (2.5\\%,\\ 97.5\\%)}\\\\')
    print('\\hline')
    
    for t in tab:
      print(t)
    
    print('\\end{tabular}')
    print('\\label{tab:real'+nu_str+noise+'}')
    print('\\end{table}')
    print('\n\n\n')
