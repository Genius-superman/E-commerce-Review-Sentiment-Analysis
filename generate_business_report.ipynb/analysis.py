# -*- coding: utf-8 -*-
"""电商评论文本情感分类 —— 完整分析流水线（含数据画像 / 建模评估 / 特征洞察 / 错分归因）"""
import json, os, re, warnings
import numpy as np
import pandas as pd
import jieba
warnings.filterwarnings('ignore')

SRC = r'C:/Users/Alienware/Desktop/情感评论分析'
OUT = r'C:/Users/Alienware/WorkBuddy/2026-10-09-16-18-09/电商评论情感分析'
ASSETS = os.path.join(OUT, 'assets')
os.makedirs(ASSETS, exist_ok=True)

R = {}  # result dict

# ============ 1. 原始数据画像 ============
raw = pd.read_csv(os.path.join(SRC, 'product_comments.csv'))
raw.columns = ['review', 'label']
raw['review'] = raw['review'].astype(str)
R['raw_total'] = int(len(raw))
R['raw_label_dist'] = {str(k): int(v) for k, v in raw['label'].value_counts().sort_index().items()}
raw['rlen'] = raw['review'].str.len()
R['raw_len'] = {
    'mean': round(float(raw['rlen'].mean()), 1),
    'median': int(raw['rlen'].median()),
    'max': int(raw['rlen'].max()),
    'min': int(raw['rlen'].min()),
    'p90': int(raw['rlen'].quantile(0.9)),
}
R['raw_dup'] = int(raw['review'].duplicated().sum())
R['raw_short'] = int((raw['rlen'] <= 5).sum())

# ============ 2. 清洗：剔除中性评论，仅保留 1/3 二分类 ============
data = raw[raw['label'] != 2].reset_index(drop=True)
LABEL_NAME = {1: '负面评论', 3: '正面评论'}
data['label_name'] = data['label'].map(LABEL_NAME)
R['clean_total'] = int(len(data))
R['clean_label_dist'] = {str(k): int(v) for k, v in data['label'].value_counts().sort_index().items()}
R['neg_ratio'] = round(float((data['label'] == 1).mean()), 4)
R['pos_ratio'] = round(float((data['label'] == 3).mean()), 4)
# 类别不平衡比
R['imbalance_ratio'] = round(float(data['label'].value_counts().max() / data['label'].value_counts().min()), 2)

# 各类别文本长度
len_by_label = data.groupby('label_name')['review'].apply(lambda s: s.str.len().mean()).round(1).to_dict()
R['len_by_label'] = len_by_label

# ============ 3. 分词 ============
with open(os.path.join(SRC, 'cn-stopwords.txt'), encoding='utf-8') as f:
    stop_words = [x.strip() for x in f.readlines() if x.strip()]
R['stopwords_n'] = len(stop_words)

data['seg_words'] = data['review'].apply(lambda x: ' '.join(jieba.cut(x)))
seg_words = data['seg_words'].tolist()
labels = data['label'].tolist()

# ============ 4. TF-IDF 向量化 ============
from sklearn.feature_extraction.text import TfidfVectorizer
tfv = TfidfVectorizer(stop_words=stop_words)
X_all = tfv.fit_transform(seg_words)
R['vocab_size'] = int(X_all.shape[1])
feature_names_all = np.array(tfv.get_feature_names_out())

# ============ 5. 卡方特征筛选 Top100 ============
from sklearn.feature_selection import SelectKBest, chi2
K = 100
selector = SelectKBest(chi2, k=K)
X_sel = selector.fit_transform(X_all, labels)
scores = selector.scores_
support = selector.get_support()
top_idx = np.argsort(scores)[::-1][:K]
top_words = [(feature_names_all[i], round(float(scores[i]), 2)) for i in top_idx]
R['top_words'] = top_words

# 判断每个Top词偏向哪个类别：比较该词在两类中的平均TF-IDF
pos_mask = np.array(labels) == 3
neg_mask = np.array(labels) == 1
word_cols = np.array([np.where(feature_names_all == w)[0][0] for w, _ in top_words])
sub = np.asarray(X_all[:, word_cols].todense())  # n_samples x 100
pos_mean = sub[pos_mask].mean(axis=0)
neg_mean = sub[neg_mask].mean(axis=0)
top_polar = []
for k, (w, s) in enumerate(top_words):
    polar = '正面' if pos_mean[k] >= neg_mean[k] else '负面'
    diff = abs(pos_mean[k] - neg_mean[k])
    top_polar.append({'word': w, 'chi2': s, 'polarity': polar, 'diff': round(float(diff), 4)})
R['top_words_polar'] = top_polar
R['top_pos_words'] = [x['word'] for x in top_polar if x['polarity'] == '正面'][:40]
R['top_neg_words'] = [x['word'] for x in top_polar if x['polarity'] == '负面'][:40]

# 高区分度（diff 最大前 20）
top_diff = sorted(top_polar, key=lambda x: -x['diff'])[:20]
R['top_discriminative'] = [{'word': x['word'], 'polarity': x['polarity'], 'diff': x['diff']} for x in top_diff]

# ============ 6. 数据集划分 + 建模 ============
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
x_train, x_test, y_train, y_test = train_test_split(X_sel, labels, random_state=0, test_size=0.25, stratify=labels)
R['train_n'] = int(x_train.shape[0])
R['test_n'] = int(x_test.shape[0])

model = MultinomialNB()
model.fit(x_train, y_train)

# 10折交叉验证 AUC
cv_auc = cross_val_score(model, x_train, y_train, cv=10, scoring='roc_auc')
R['cv_auc_mean'] = round(float(np.mean(cv_auc)), 4)
R['cv_auc_std'] = round(float(np.std(cv_auc)), 4)
R['cv_auc_folds'] = [round(float(x), 4) for x in cv_auc]

# 交叉验证准确率
cv_acc = cross_val_score(model, x_train, y_train, cv=10, scoring='accuracy')
R['cv_acc_mean'] = round(float(np.mean(cv_acc)), 4)
R['cv_acc_std'] = round(float(np.std(cv_acc)), 4)

# ============ 7. 预测与评估 ============
y_pred = model.predict(x_test)
from sklearn.metrics import (confusion_matrix, precision_score, recall_score, f1_score,
                             accuracy_score, classification_report)
cm = confusion_matrix(y_test, y_pred)
R['confusion_matrix'] = cm.tolist()
R['cm_labels'] = [1, 3]
R['accuracy'] = round(float(accuracy_score(y_test, y_pred)), 4)

# 每类指标（labels=1,3）
p = precision_score(y_test, y_pred, labels=[1, 3], average=None, zero_division=0)
r = recall_score(y_test, y_pred, labels=[1, 3], average=None, zero_division=0)
f = f1_score(y_test, y_pred, labels=[1, 3], average=None, zero_division=0)
R['per_class'] = {
    'negative': {'precision': round(float(p[0]), 4), 'recall': round(float(r[0]), 4), 'f1': round(float(f[0]), 4)},
    'positive': {'precision': round(float(p[1]), 4), 'recall': round(float(r[1]), 4), 'f1': round(float(f[1]), 4)},
}
R['macro_f1'] = round(float(f1_score(y_test, y_pred, average='macro')), 4)
R['weighted_f1'] = round(float(f1_score(y_test, y_pred, average='weighted')), 4)

# 样本外预测分布（模型偏置诊断）
R['pred_dist'] = {'1': int((y_pred == 1).sum()), '3': int((y_pred == 3).sum())}

# ============ 8. 错分样本分析 ============
test_reviews = data['review'].values
test_labels = np.array(labels)
# 还原测试集原始文本索引
tr_idx, te_idx = train_test_split(np.arange(len(data)), random_state=0, test_size=0.25, stratify=labels)
err_mask = y_pred != np.array(y_test)
err_idx = te_idx[err_mask]
err_true = test_labels[err_idx]
err_pred = y_pred[err_mask]
err_reviews = data['review'].values[err_idx]

err_samples = []
for rev, t, pr in zip(err_reviews, err_true, err_pred):
    err_samples.append({'review': rev, 'true': LABEL_NAME[t], 'pred': LABEL_NAME[pr], 'len': len(str(rev))})
R['error_total'] = int(len(err_samples))
R['error_rate'] = round(float(len(err_samples) / len(y_test)), 4)
# 按错误类型
pos2neg = [e for e in err_samples if e['true'] == '负面评论' and e['pred'] == '正面评论']  # FN
neg2pos = [e for e in err_samples if e['true'] == '正面评论' and e['pred'] == '负面评论']  # FP
R['neg_as_pos_n'] = len(pos2neg)  # 真正的负面被判为正面（漏报，业务风险高）
R['pos_as_neg_n'] = len(neg2pos)
R['err_len_mean'] = round(float(np.mean([e['len'] for e in err_samples])), 1)

# 给错分样本打"误判原因"标签（启发式）
IRONY_KW = ['呵呵', '哈哈', '真是', '醉了', '服了', '厉害', '优秀', '牛', '绝了', '无语', '终于', '也算', '还行吧', '的确', '确实']
for e in err_samples:
    rev = e['review']
    reasons = []
    if len(rev) <= 8:
        reasons.append('短句/信息量不足')
    if any(k in rev for k in IRONY_KW):
        reasons.append('疑似反讽/口语歧义')
    if ('不' in rev or '没' in rev) and e['true'] == '正面评论':
        reasons.append('否定词歧义')
    if not reasons:
        reasons.append('语义隐含/无显式情感词')
    e['reason'] = '、'.join(reasons)
R['error_samples'] = err_samples[:300]
# 原因分布
from collections import Counter
rc = Counter()
for e in err_samples:
    for x in e['reason'].split('、'):
        rc[x] += 1
R['error_reason_dist'] = dict(rc.most_common())

# 误判样本（负面→正面）最典型示例
R['fn_examples'] = [e['review'] for e in pos2neg][:8]
R['fp_examples'] = [e['review'] for e in neg2pos][:8]

# ============ 9. 词频统计（全量 & 分极性） ============
def top_freq(corpus_words, n=30):
    c = Counter()
    for ws in corpus_words:
        for w in ws.split():
            if len(w) > 1 and w not in stop_words:
                c[w] += 1
    return c.most_common(n)

R['freq_all'] = top_freq(seg_words, 40)
R['freq_neg'] = top_freq(data[data['label'] == 1]['seg_words'].tolist(), 30)
R['freq_pos'] = top_freq(data[data['label'] == 3]['seg_words'].tolist(), 30)

# 评论长度分布直方
R['len_hist'] = np.histogram(data['review'].str.len().clip(upper=200), bins=20)[0].tolist()

with open(os.path.join(OUT, 'stats.json'), 'w', encoding='utf-8') as f:
    json.dump(R, f, ensure_ascii=False, indent=2)

print('=== 数据画像 ===')
print('原始样本:', R['raw_total'], '| 清洗后:', R['clean_total'])
print('原始标签分布:', R['raw_label_dist'])
print('清洗后分布:', R['clean_label_dist'], '| 不平衡比:', R['imbalance_ratio'])
print('词表规模:', R['vocab_size'], '| 停用词:', R['stopwords_n'])
print('训练/测试:', R['train_n'], '/', R['test_n'])
print('CV AUC:', R['cv_auc_mean'], '±', R['cv_auc_std'])
print('CV Acc:', R['cv_acc_mean'])
print('测试集准确率:', R['accuracy'])
print('混淆矩阵:', R['confusion_matrix'])
print('每类:', R['per_class'])
print('错分:', R['error_total'], '错误率:', R['error_rate'])
print('负面→正面(漏报):', R['neg_as_pos_n'], '| 正面→负面(误报):', R['pos_as_neg_n'])
print('错因分布:', R['error_reason_dist'])
print('Top20区分词:', [(x['word'], x['polarity']) for x in R['top_discriminative']])
print('OK -> stats.json')
