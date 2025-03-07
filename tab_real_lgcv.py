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

nu_str='_100_'
data_sets=sorted(list(set(map(lambda s: re.split('/|_',s)[3],glob(REAL_DATA+'/lgcv_*')))))

lgcv_dict={}
for data in data_sets:
  lgcv_dict[data]={}
  seeds=list(map(lambda s: s.split('_')[-1],glob(REAL_DATA+'/lgcv_'+data+'_*')))
  
  for seed in seeds:
    fi=open(REAL_DATA+'/lgcv_'+data+nu_str+seed,'rb')
    lgcv_dict_seed=pickle.load(fi)
    fi.close()
    for kr_name in lgcv_dict_seed.keys():
      if not kr_name in lgcv_dict[data].keys():
        lgcv_dict[data][kr_name]={}
      for cv_type in lgcv_dict_seed[kr_name].keys():
        if not cv_type in lgcv_dict[data][kr_name].keys():
          lgcv_dict[data][kr_name][cv_type]={}
        for metric_name in lgcv_dict_seed[kr_name][cv_type].keys():
          if not metric_name in lgcv_dict[data][kr_name][cv_type].keys():
            lgcv_dict[data][kr_name][cv_type][metric_name]=[]
          lgcv_dict[data][kr_name][cv_type][metric_name].append(lgcv_dict_seed[kr_name][cv_type][metric_name])

tab0=[]
tab1=[]

fun_titles={'kpr': 'K$\\ell_\\infty$R', 'krr': 'KRR'}
cv_titles={'gcv': '& GCV', 'loocv': 'LOOCV'}
data_titles={'airfoil': '\\makecell[l]{Airfoil Sound\\\\Pressure}', 'house': '\\makecell[l]{California\\\\House Values}', 'temp': '\\makecell[l]{U.K.\\\\Temperature}', 'steel': '\\makecell[l]{Steel Energy\\\\Consumption}', 'super': '\\makecell[l]{Superconductor\\\\Critical\\\\Temperature}'}

seen_data=[]
old_fun=''
for data in data_sets:
  for kr_fun in ['kpr','krr']:
    for cv_type in ['loocv', 'gcv']:
      q1_r2=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['r2'],Q1)
      q2_r2=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['r2'],0.5)
      q3_r2=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['r2'],1-Q1)
      q1_time=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['time'],Q1)
      q2_time=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['time'],0.5)
      q3_time=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['time'],1-Q1)
      q1_sig=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['sigma'],Q1)
      q2_sig=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['sigma'],0.5)
      q3_sig=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['sigma'],1-Q1)
      q1_lab=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['lbda'],Q1)
      q2_lab=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['lbda'],0.5)
      q3_lab=np.nanquantile(lgcv_dict[data][kr_fun][cv_type]['lbda'],1-Q1)
      
      if not data in seen_data:
        seen_data.append(data)
        data_str='\\multirow{4}{*}{'+data_titles[data.split('-')[0]]+'}\n'
      else:
        data_str=''
      if kr_fun!=old_fun:
        if kr_fun=='krr':
          data_str+='\\cline{2-5}\n'
        data_str+='& \\multirow{2}{*}{'+fun_titles[kr_fun]+'}\n'
        old_fun=kr_fun
      data_str+=f'& {cv_titles[cv_type]:<5} & ${q2_time:#.3g},\\ ({q1_time:#.3g}, {q3_time:#.3g})$ & ${q2_r2:<5.2f},\\ ({q1_r2:<7.2f}, {q3_r2:<5.2f})$ \\\\'.replace('.,',',').replace('.)',')')
      if kr_fun =='krr' and cv_type=='gcv':
        data_str+='\n\\hline'
      if data[-1:]=='0':
        tab0.append(data_str)
      elif data[-1:]=='1':
        tab1.append(data_str)


for tab, noise in zip([tab0, tab1],['0','1']):
  if noise=='1':
    noise_str='with'
  elif noise=='0':
    noise_str='without'
  print('\\begin{table}')
  print('\\caption{The 2.5th, 50th, and 97.5th percentiles of computation time and test $R^2$ for K$\\ell_\\infty$R in combination with LOOCV and GCV on the different data sets, \\textbf{'+noise_str+'} amplified outliers. Compared to 10-fold cross validation LOOCV and GCV perform faster, but the cost of decreased predictive performance, especially for K$\\ell_\\infty$R.}')

  print('\\center')
  print('\\fontsize{7.9}{9.5}\\selectfont')
  print('\\begin{tabular}{l|l|l|l|l}')
  print('\\hline')
  print('Data & Method & \\makecell[l]{Type of\\\\Cross-\\\\Validation} & \\makecell[l]{Computation\\\\Time [s]\\\\50\\%,\\ (2.5\\%,\\ 97.5\\%)} & \\makecell[l]{Test $R^2$\\\\50\\%,\\ (2.5\\%,\\ 97.5\\%)}\\\\')
  print('\\hline')
  
  for t in tab:
    print(t)
  
  print('\\end{tabular}')
  print('\\label{tab:real_lgcv'+nu_str+noise+'}')
  print('\\end{table}')
  print('\n\n\n')


