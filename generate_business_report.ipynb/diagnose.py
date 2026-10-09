# -*- coding: utf-8 -*-
"""深度诊断：词袋特征覆盖率、多数类偏置、Top100完整清单"""
import json, os, warnings
import numpy as np, pandas as pd, jieba
from collections import Counter
warnings.filterwarnings('ignore')

SRC = r'C:/Users/Alienware/Desktop/情感评论分析'
OUT = r'C:/Users/Alienware/WorkBuddy/2026-10-09-16-18-09/电商评论情感分析'
R = json.load(open(os.path.join(OUT, 'stats.json'), encoding='utf-8'))

USER_WORDS = ['手机','电池','待机时间','拍照','屏幕','音效','外观','外形','速度','运行','流畅',
              '性价比','物流','客服','包装','手感','分辨率','充电','续航','性能','处理器','系统']
for w in USER_WORDS: jieba.add_word(w)
with open(os.path.join(SRC, 'cn-stopwords.txt'), encoding='utf-8') as f:
    stop_words = [x.strip() for x in f.readlines() if x.strip()]

raw = pd.read_csv(os.path.join(SRC, 'product_comments.csv')); raw.columns=['review','label']
raw['review'] = raw['review'].astype(str)
data = raw[raw['label']!=2].reset_index(drop=True)
data['seg'] = data['review'].apply(lambda x: ' '.join(jieba.cut(x)))

from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(stop_words=stop_words)
X = tfv.fit_transform(data['seg'].tolist())
names = np.array(tfv.get_feature_names_out())

from sklearn.feature_selection import SelectKBest, chi2
sel = SelectKBest(chi2, k=100).fit(X, data['label'].tolist())
mask = sel.get_support()
top100 = names[mask]
R['top100_full'] = top100.tolist()

# 每条评论命中的 top100 特征词数量
Xsel = X[:, mask]
hits = np.asarray((Xsel > 0).sum(axis=1)).ravel()
data['hits'] = hits
neg_hits = data[data['label']==1]['hits']
pos_hits = data[data['label']==3]['hits']
R['hits_neg_mean'] = round(float(neg_hits.mean()), 2)
R['hits_pos_mean'] = round(float(pos_hits.mean()), 2)

# 零覆盖评论比例（完全不含任何Top100特征 → 只能靠先验预测）
zero_mask = hits == 0
R['zero_cov_total'] = int(zero_mask.sum())
R['zero_cov_ratio'] = round(float(zero_mask.mean()), 4)
R['zero_cov_neg'] = int((zero_mask & (data['label']==1)).sum())
R['zero_cov_pos'] = int((zero_mask & (data['label']==3)).sum())
R['zero_cov_neg_ratio_in_neg'] = round(float((zero_mask & (data['label']==1)).sum() / (data['label']==1).sum()), 4)

# 覆盖率分箱 vs 是否负面
data['cov_bin'] = pd.cut(hits, [-1,0,1,2,3,100], labels=['0','1','2','3','4+'])
tab = data.groupby('cov_bin', observed=False).apply(
    lambda d: pd.Series({'n': len(d), 'neg_rate': round(float((d['label']==1).mean()),3)}))
R['coverage_neg_rate'] = {str(k): {'n': int(v['n']), 'neg_rate': float(v['neg_rate'])} for k,v in tab.iterrows()}

# 极化词统计
marg = np.asarray(Xsel.sum(axis=0)).ravel()
order = np.argsort(marg)[::-1]
R['top100_by_weight'] = [(top100[i], round(float(marg[i]),1)) for i in order[:30]]

json.dump(R, open(os.path.join(OUT,'stats.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)

print('Top100特征词覆盖率诊断')
print('负面评论平均命中Top100特征数:', R['hits_neg_mean'])
print('正面评论平均命中Top100特征数:', R['hits_pos_mean'])
print('零覆盖评论数:', R['zero_cov_total'], f"({R['zero_cov_ratio']:.1%})")
print('  其中负面:', R['zero_cov_neg'], '正面:', R['zero_cov_pos'])
print('负面类中零覆盖占比:', R['zero_cov_neg_ratio_in_neg'])
print('覆盖率 vs 负面率:', R['coverage_neg_rate'])
print()
print('Top100 完整清单（前40）:', R['top100_full'][:40])
print()
print('=== 含"降价"的负面评论占比 ===')
d2 = data[data['label']==1]
print('降价类投诉:', d2['review'].str.contains('降价|掉价|差价|价保|退差').sum(), '/', len(d2))
print('售后类投诉:', d2['review'].str.contains('售后|退货|换货|返修|维修').sum())
print('客服类投诉:', d2['review'].str.contains('客服').sum())
