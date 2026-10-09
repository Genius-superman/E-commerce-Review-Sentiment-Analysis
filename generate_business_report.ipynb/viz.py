# -*- coding: utf-8 -*-
"""生成专业图表：词云 / 混淆矩阵 / 类别分布 / PRF1 / 特征词 / 错因 / 长度分布"""
import json, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
warnings.filterwarnings('ignore')

SRC = r'C:/Users/Alienware/Desktop/情感评论分析'
OUT = r'C:/Users/Alienware/WorkBuddy/2026-10-09-16-18-09/电商评论情感分析'
ASSETS = os.path.join(OUT, 'assets')
os.makedirs(ASSETS, exist_ok=True)

R = json.load(open(os.path.join(OUT, 'stats.json'), encoding='utf-8'))

# ---- 中文字体 ----
for fp in [r'C:/Windows/Fonts/msyh.ttc', r'C:/Windows/Fonts/msyhbd.ttc', r'C:/Windows/Fonts/simhei.ttf']:
    if os.path.exists(fp):
        font_manager.fontManager.addfont(fp)
        CN_FONT = font_manager.FontProperties(fname=fp).get_name()
        break
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 12

# 配色（中国股市惯例：涨=红、跌=绿；此处正面=红/暖色，负面=绿/冷色）
RED = '#c0392b'
GREEN = '#27ae60'
NEG_COLOR = '#e74c3c'   # 负面 / 差评 = 红色（预警）
POS_COLOR = '#2e9e6b'   # 正面 / 好评 = 绿色
BLUE = '#2c6fbb'
INK = '#1f2937'
GRID = '#e5e7eb'

def save(fig, name):
    p = os.path.join(ASSETS, name)
    fig.savefig(p, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('saved', name)

# ========== 1. 词云 ==========
from wordcloud import WordCloud
def make_wc(freq, name, cmap, bg='white'):
    wc = WordCloud(font_path=r'C:/Windows/Fonts/msyh.ttc', width=900, height=520,
                   background_color=bg, colormap=cmap, max_words=120,
                   prefer_horizontal=0.92, margin=4, random_state=42)
    wc.generate_from_frequencies(dict(freq))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.imshow(wc, interpolation='bilinear'); ax.axis('off')
    save(fig, name)

make_wc(R['freq_all'][:120], 'wc_all.png', 'viridis')
make_wc(R['freq_neg'], 'wc_neg.png', 'Greens')
make_wc(R['freq_pos'], 'wc_pos.png', 'Reds')

# ========== 2. 混淆矩阵（修正版 2x2） ==========
cm = np.array(R['confusion_matrix'])
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
labels_cn = ['负面', '正面']
for ax, mat, title, fmt in [
    (axes[0], cm, '混淆矩阵（样本量）', 'd'),
    (axes[1], cm.astype(float) / cm.sum(axis=1, keepdims=True), '混淆矩阵（按真实类别归一化）', '.2f')]:
    im = ax.imshow(mat, cmap='Blues', vmin=0)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([f'预测\n{x}' for x in labels_cn]); ax.set_yticklabels([f'真实\n{x}' for x in labels_cn])
    ax.set_title(title, fontsize=14, pad=12, color=INK, fontweight='bold')
    for i in range(2):
        for j in range(2):
            v = mat[i, j]
            txt = f'{int(v)}' if fmt == 'd' else f'{v:.1%}'
            ax.text(j, i, txt, ha='center', va='center', fontsize=16,
                    color='white' if v > mat.max() * 0.55 else INK, fontweight='bold')
    ax.set_xlabel('预测标签'); ax.set_ylabel('真实标签')
save(fig, 'confusion_matrix_fixed.png')

# ========== 3. 类别分布 ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
raw = R['raw_label_dist']
names3 = ['负面(1)', '中性(2)', '正面(3)']
vals3 = [raw['1'], raw['2'], raw['3']]
cols3 = [NEG_COLOR, '#b8bcc4', POS_COLOR]
b = axes[0].bar(names3, vals3, color=cols3, width=0.55)
axes[0].set_title('原始数据三类情感分布', fontsize=14, fontweight='bold', color=INK)
for rect, v in zip(b, vals3):
    axes[0].text(rect.get_x() + rect.get_width()/2, v + 80, f'{v}\n({v/sum(vals3):.1%})',
                 ha='center', fontsize=11, color=INK)
axes[0].set_ylim(0, max(vals3)*1.2); axes[0].set_ylabel('评论数')
axes[0].grid(axis='y', color=GRID); axes[0].set_axisbelow(True)
for s in ['top', 'right']: axes[0].spines[s].set_visible(False)

cl = R['clean_label_dist']
names2 = ['负面评论', '正面评论']
vals2 = [cl['1'], cl['3']]
w, _, at = axes[1].pie(vals2, labels=names2, autopct='%1.1f%%', startangle=90,
                        colors=[NEG_COLOR, POS_COLOR], textprops={'fontsize': 12, 'color': INK},
                        wedgeprops={'width': 0.42, 'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title(f'清洗后二分类分布（去掉中性，不平衡比 {R["imbalance_ratio"]}:1）',
                  fontsize=13, fontweight='bold', color=INK)
save(fig, 'class_dist.png')

# ========== 4. 精确率/召回率/F1 对比 ==========
pc = R['per_class']
metrics = ['precision', 'recall', 'f1']
mnames = ['精确率 Precision', '召回率 Recall', 'F1 分数']
x = np.arange(len(metrics)); width = 0.35
fig, ax = plt.subplots(figsize=(9.5, 5))
neg_vals = [pc['negative'][m] for m in metrics]
pos_vals = [pc['positive'][m] for m in metrics]
b1 = ax.bar(x - width/2, neg_vals, width, label='负面评论', color=NEG_COLOR)
b2 = ax.bar(x + width/2, pos_vals, width, label='正面评论', color=POS_COLOR)
for bars in [b1, b2]:
    for rect in bars:
        ax.text(rect.get_x()+rect.get_width()/2, rect.get_height()+0.015, f'{rect.get_height():.2f}',
                ha='center', fontsize=11, fontweight='bold', color=INK)
ax.set_xticks(x); ax.set_xticklabels(mnames, fontsize=12)
ax.set_ylim(0, 1.15); ax.set_ylabel('得分'); ax.legend(frameon=False, fontsize=12, loc='lower right')
ax.set_title(f'各类别分类性能对比  |  加权F1 = {R["weighted_f1"]:.3f}，宏平均F1 = {R["macro_f1"]:.3f}',
             fontsize=13, fontweight='bold', color=INK)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
save(fig, 'prf1.png')

# ========== 5. Top 区分度特征词（发散条形） ==========
td = R['top_discriminative'][:18]
words = [x['word'] for x in td][::-1]
diffs = [x['diff'] for x in td][::-1]
pols = [x['polarity'] for x in td][::-1]
colors = [POS_COLOR if p == '正面' else NEG_COLOR for p in pols]
fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(words, diffs, color=colors)
for rect, p in zip(bars, pols):
    ax.text(rect.get_width()+0.0008, rect.get_y()+rect.get_height()/2, p,
            va='center', fontsize=9, color=INK)
ax.set_xlabel('两类平均 TF-IDF 差值（区分度，越大越有判别力）')
ax.set_title('卡方筛选 Top100 中区分度最强的 18 个特征词', fontsize=13, fontweight='bold', color=INK)
ax.grid(axis='x', color=GRID); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
save(fig, 'top_features.png')

# ========== 6. 错因分布 ==========
er = R['error_reason_dist']
fig, ax = plt.subplots(figsize=(9, 4.6))
ks = list(er.keys()); vs = list(er.values())
colors_e = ['#8e6fbf', '#e08a3c', '#5b9bd5', '#7f8c8d'][:len(ks)]
bars = ax.barh(ks[::-1], vs[::-1], color=colors_e[::-1])
for rect in bars:
    ax.text(rect.get_width()+3, rect.get_y()+rect.get_height()/2, str(int(rect.get_width())),
            va='center', fontsize=11, fontweight='bold', color=INK)
ax.set_xlabel('错分样本数（同一评论可能命中多个原因）')
ax.set_title(f'错分样本归因分析（共 {R["error_total"]} 条，占比 {R["error_rate"]:.1%}）',
             fontsize=13, fontweight='bold', color=INK)
ax.grid(axis='x', color=GRID); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
save(fig, 'error_reasons.png')

# ========== 7. 评论长度分布 ==========
hist = np.array(R['len_hist'])
edges = np.linspace(0, 200, len(hist)+1)
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.bar(edges[:-1], hist, width=(edges[1]-edges[0])*0.92, color=BLUE, align='edge')
ax.set_xlabel('评论字符长度（超长部分已截断至 200）'); ax.set_ylabel('评论数')
ax.set_title('评论长度分布：平均 {:} 字符，中位数 {} 字符'.format(R['raw_len']['mean'], R['raw_len']['median']),
             fontsize=13, fontweight='bold', color=INK)
ax.grid(axis='y', color=GRID); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
save(fig, 'length_dist.png')

# ========== 8. 交叉验证 10 折 AUC 稳定性 ==========
folds = R['cv_auc_folds']
fig, ax = plt.subplots(figsize=(9, 4.4))
xs = np.arange(1, len(folds)+1)
ax.plot(xs, folds, marker='o', color=BLUE, linewidth=2, markersize=7, label='各折 AUC')
ax.axhline(R['cv_auc_mean'], color=RED, linestyle='--', linewidth=1.6,
           label=f'均值 {R["cv_auc_mean"]:.4f}')
ax.fill_between(xs, R['cv_auc_mean']-R['cv_auc_std'], R['cv_auc_mean']+R['cv_auc_std'],
                color=BLUE, alpha=0.10)
ax.set_xticks(xs); ax.set_xlabel('交叉验证折次'); ax.set_ylabel('AUC')
ax.set_ylim(min(folds)-0.01, max(folds)+0.01)
ax.set_title(f'10 折交叉验证 AUC 稳定性（{R["cv_auc_mean"]:.4f} ± {R["cv_auc_std"]:.4f}）',
             fontsize=13, fontweight='bold', color=INK)
ax.legend(frameon=False); ax.grid(color=GRID); ax.set_axisbelow(True)
for s in ['top', 'right']: ax.spines[s].set_visible(False)
save(fig, 'cv_auc.png')

print('ALL CHARTS DONE')
