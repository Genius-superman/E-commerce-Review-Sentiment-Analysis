# -*- coding: utf-8 -*-
"""补充图表：特征覆盖率诊断 + 模型对比实验"""
import json, os
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT=r'C:/Users/Alienware/WorkBuddy/2026-10-09-16-18-09/电商评论情感分析'
A=os.path.join(OUT,'assets')
R=json.load(open(os.path.join(OUT,'stats.json'),encoding='utf-8'))
for fp in [r'C:/Windows/Fonts/msyh.ttc']:
    if os.path.exists(fp): font_manager.fontManager.addfont(fp)
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei']
plt.rcParams['axes.unicode_minus']=False
INK='#1f2937'; GRID='#e5e7eb'; RED='#e74c3c'; GRN='#2e9e6b'; BLU='#2c6fbb'; AMB='#e08a3c'
# --- 1. 覆盖率 vs 负面率 ---
cov=R['coverage_neg_rate']
ks=['0','1','2','3','4+']
ns=[cov[k]['n'] for k in ks]; nr=[cov[k]['neg_rate'] for k in ks]
fig,ax1=plt.subplots(figsize=(9.5,5))
b=ax1.bar(ks,ns,color='#cbd5e1',width=0.6,label='评论数（左轴）')
for r,v in zip(b,ns): ax1.text(r.get_x()+r.get_width()/2,v+60,str(v),ha='center',fontsize=10,color=INK)
ax1.set_xlabel('该评论命中的 Top100 特征词数量'); ax1.set_ylabel('评论数'); ax1.set_ylim(0,max(ns)*1.18)
ax1.grid(axis='y',color=GRID); ax1.set_axisbelow(True)
for s in ['top','right']: ax1.spines[s].set_visible(False)
ax2=ax1.twinx()
ax2.plot(ks,[x*100 for x in nr],color=RED,marker='o',linewidth=2.5,markersize=8,label='负面评论占比（右轴）')
for i,(k,v) in enumerate(zip(ks,nr)):
    ax2.text(i,v*100+3,f'{v:.0%}',ha='center',fontsize=11,color=RED,fontweight='bold')
ax2.set_ylabel('负面评论占比 (%)',color=RED); ax2.tick_params(axis='y',colors=RED); ax2.set_ylim(0,100)
h1,l1=ax1.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax1.legend(h1+h2,l1+l2,frameon=False,loc='upper center',fontsize=11)
ax1.set_title('特征覆盖率诊断：命中 Top100 词越少，越可能是负面评论\n'
              f"（负面评论平均仅命中 {R['hits_neg_mean']} 个词，正面评论命中 {R['hits_pos_mean']} 个）",
              fontsize=13,fontweight='bold',color=INK)
fig.tight_layout(); fig.savefig(os.path.join(A,'coverage_diag.png'),dpi=200,bbox_inches='tight',facecolor='white'); plt.close(fig)
print('saved coverage_diag.png')

# --- 2. 模型对比实验 ---
ex=R['experiments']
def short(n):
    return (n.replace('朴素贝叶斯 + ','NB+').replace('卡方','χ²').replace('（原方案）','')
             .replace(' + 类别平衡权重','+平衡').replace('ComplementNB + ','CNB+')
             .replace('（适配不平衡）','').replace('全特征（不筛选）','全特征'))
names=[short(e['name']) for e in ex]
acc=[e['accuracy'] for e in ex]; nrec=[e['neg_recall'] for e in ex]; f1m=[e['macro_f1'] for e in ex]
x=np.arange(len(names)); w=0.26
fig,ax=plt.subplots(figsize=(13.5,6))
ax.bar(x-w,acc,w,label='准确率',color=BLU)
ax.bar(x,nrec,w,label='负面类召回率',color=RED)
ax.bar(x+w,f1m,w,label='宏平均F1',color=AMB)
for i in range(len(names)):
    for off,v,c in [(-w,acc[i],BLU),(0,nrec[i],RED),(w,f1m[i],AMB)]:
        ax.text(i+off,v+0.012,f'{v:.3f}',ha='center',fontsize=8.6,color=INK,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(names,fontsize=10)
ax.set_ylim(0,1.12); ax.set_ylabel('得分'); ax.legend(frameon=False,fontsize=11,ncol=3,loc='upper center')
ax.set_title('模型改进对比实验：特征数量与类别平衡策略的效果',fontsize=14,fontweight='bold',color=INK)
ax.grid(axis='y',color=GRID); ax.set_axisbelow(True)
for s in ['top','right']: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(A,'experiments.png'),dpi=200,bbox_inches='tight',facecolor='white'); plt.close(fig)
print('saved experiments.png')
print('DONE')
