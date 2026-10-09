# -*- coding: utf-8 -*-
"""模型对比实验：不同特征数 / 不同算法 / 类别权重平衡，验证改进方向"""
import json, os, warnings, time
import numpy as np, pandas as pd, jieba
warnings.filterwarnings('ignore')

SRC = r'C:/Users/Alienware/Desktop/情感评论分析'
OUT = r'C:/Users/Alienware/WorkBuddy/2026-10-09-16-18-09/电商评论情感分析'
R = json.load(open(os.path.join(OUT,'stats.json'),encoding='utf-8'))

with open(os.path.join(SRC,'cn-stopwords.txt'),encoding='utf-8') as f:
    stop_words=[x.strip() for x in f.readlines() if x.strip()]
raw=pd.read_csv(os.path.join(SRC,'product_comments.csv')); raw.columns=['review','label']
raw['review']=raw['review'].astype(str)
data=raw[raw['label']!=2].reset_index(drop=True)
data['seg']=data['review'].apply(lambda x:' '.join(jieba.cut(x)))
y=np.array(data['label'].tolist())

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

tfv=TfidfVectorizer(stop_words=stop_words)
X=tfv.fit_transform(data['seg'].tolist())

def experiment(name, clf, k, balanced=False):
    Xk = SelectKBest(chi2, k=k).fit_transform(X, y) if k else X
    xtr,xte,ytr,yte=train_test_split(Xk,y,random_state=0,test_size=0.25,stratify=y)
    if balanced:
        sw=compute_sample_weight('balanced', ytr)
        clf.fit(xtr,ytr,sample_weight=sw)
    else:
        clf.fit(xtr,ytr)
    pred=clf.predict(xte)
    acc=accuracy_score(yte,pred)
    neg_r=recall_score(yte,pred,pos_label=1,zero_division=0)   # 负面召回
    neg_p=precision_score(yte,pred,pos_label=1,zero_division=0)
    f1m=f1_score(yte,pred,average='macro')
    f1w=f1_score(yte,pred,average='weighted')
    nfeat = Xk.shape[1]
    return {'name':name,'n_features':int(nfeat),'accuracy':round(float(acc),4),
            'neg_recall':round(float(neg_r),4),'neg_precision':round(float(neg_p),4),
            'macro_f1':round(float(f1m),4),'weighted_f1':round(float(f1w),4)}

results=[]
results.append(experiment('朴素贝叶斯 + 卡方Top100（原方案）', MultinomialNB(), 100))
results.append(experiment('朴素贝叶斯 + 卡方Top100 + 类别平衡权重', MultinomialNB(), 100, True))
results.append(experiment('朴素贝叶斯 + 卡方Top500', MultinomialNB(), 500))
results.append(experiment('朴素贝叶斯 + 卡方Top500 + 类别平衡权重', MultinomialNB(), 500, True))
results.append(experiment('朴素贝叶斯 + 卡方Top1000', MultinomialNB(), 1000))
results.append(experiment('朴素贝叶斯 + 全特征（不筛选）', MultinomialNB(), None))
results.append(experiment('ComplementNB + 卡方Top500（适配不平衡）', ComplementNB(), 500))

for r in results:
    print(f"{r['name']:<38} 特征{r['n_features']:>6}  Acc={r['accuracy']:.4f}  负面召回={r['neg_recall']:.4f}  宏F1={r['macro_f1']:.4f}  加权F1={r['weighted_f1']:.4f}")

R['experiments']=results
json.dump(R, open(os.path.join(OUT,'stats.json'),'w',encoding='utf-8'), ensure_ascii=False, indent=2)
print('SAVED')
