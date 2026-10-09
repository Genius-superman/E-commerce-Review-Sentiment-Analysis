# -*- coding: utf-8 -*-
"""补充：优化分词（自定义词典）后重建词云；基于卡方 Top100 构建极性强弱词云"""
import json, os, warnings
import numpy as np, pandas as pd, jieba
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud
from collections import Counter
warnings.filterwarnings('ignore')

SRC = r'C:/Users/Alienware/Desktop/情感评论分析'
OUT = r'C:/Users/Alienware/WorkBuddy/2026-10-09-16-18-09/电商评论情感分析'
ASSETS = os.path.join(OUT, 'assets')
R = json.load(open(os.path.join(OUT, 'stats.json'), encoding='utf-8'))

# 自定义词典：保护"手机"等易被切碎的电商高频词
USER_WORDS = ['手机', '电池', '待机时间', '拍照', '屏幕', '音效', '外观', '外形', '速度',
              '运行', '流畅', '性价比', '物流', '客服', '包装', '手感', '分辨率', '充电',
              '续航', '性能', '处理器', '系统', '苹果', '华为', '小米', '京东', '质量',
              '效果', '清晰', '游戏', '信号', '散热', '很卡', '发热', '快递', '售后']
for w in USER_WORDS:
    jieba.add_word(w)

with open(os.path.join(SRC, 'cn-stopwords.txt'), encoding='utf-8') as f:
    stop_words = set(x.strip() for x in f.readlines() if x.strip())
stop_words |= {'手机', '还是', '就是', '一个', '没有', '这个', '感觉', '可以', '东西', '值得', '真的', '特别'}

raw = pd.read_csv(os.path.join(SRC, 'product_comments.csv'))
raw.columns = ['review', 'label']
raw['review'] = raw['review'].astype(str)
data = raw[raw['label'] != 2].reset_index(drop=True)

def tokenize(t):
    return [w for w in jieba.cut(t) if len(w) > 1 and w not in stop_words]

data['tokens'] = data['review'].apply(tokenize)

# 整体词频（优化分词后）
def build_freq(df):
    c = Counter()
    for toks in df['tokens']:
        c.update(toks)
    return c

freq_all = build_freq(data)
freq_neg = build_freq(data[data['label'] == 1])
freq_pos = build_freq(data[data['label'] == 3])
print('优化后 Top15 全量:', freq_all.most_common(15))
R['freq_all_v2'] = freq_all.most_common(50)
R['freq_neg_v2'] = freq_neg.most_common(30)
R['freq_pos_v2'] = freq_pos.most_common(30)

for fp in [r'C:/Windows/Fonts/msyh.ttc']:
    if os.path.exists(fp): font_manager.fontManager.addfont(fp)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
FONT = r'C:/Windows/Fonts/msyh.ttc'

def wc(freq, path, cmap, title, titlecolor='#1f2937'):
    w = WordCloud(font_path=FONT, width=1000, height=560, background_color='white',
                  colormap=cmap, max_words=130, prefer_horizontal=0.95, margin=5, random_state=7)
    w.generate_from_frequencies(dict(freq))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.imshow(w, interpolation='bilinear'); ax.axis('off')
    ax.set_title(title, fontsize=15, fontweight='bold', color=titlecolor, pad=10)
    fig.savefig(os.path.join(ASSETS, path), dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig); print('saved', path)

wc(freq_all, 'wc_all.png', 'viridis', '全网评论高频词云（优化分词后）')
wc(freq_neg, 'wc_neg.png', 'Reds', '负面评论高频词', '#c0392b')
wc(freq_pos, 'wc_pos.png', 'Greens', '正面评论高频词', '#1e7a4d')

# 基于卡方 Top100 的极性词云（词大小 = chi2 得分）
top_polar = R['top_words_polar']
pos_d = {x['word']: x['chi2'] for x in top_polar if x['polarity'] == '正面'}
neg_d = {x['word']: x['chi2'] for x in top_polar if x['polarity'] == '负面'}
print('正面特征词 n=', len(pos_d), '负面特征词 n=', len(neg_d))
wc(pos_d, 'wc_chi2_pos.png', 'Greens', f'卡方 Top100 中的 {len(pos_d)} 个"正面倾向"高区分度词（大小=卡方得分）', '#1e7a4d')
wc(neg_d, 'wc_chi2_neg.png', 'Reds', f'卡方 Top100 中的 {len(neg_d)} 个"负面倾向"高区分度词（大小=卡方得分）', '#c0392b')

# 极性强弱对比条形（正/负各 Top12 by chi2）
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for ax, d, color, name in [(axes[0], neg_d, '#e74c3c', '负面倾向'),
                            (axes[1], pos_d, '#2e9e6b', '正面倾向')]:
    items = sorted(d.items(), key=lambda x: x[1])[-12:]
    ws = [i[0] for i in items]; vs = [i[1] for i in items]
    ax.barh(ws, vs, color=color)
    ax.set_title(f'{name}词 · 卡方得分 Top12', fontsize=13, fontweight='bold', color='#1f2937')
    ax.grid(axis='x', color='#e5e7eb'); ax.set_axisbelow(True)
    for s in ['top', 'right']: ax.spines[s].set_visible(False)
    for rect in ax.patches:
        ax.text(rect.get_width()+max(vs)*0.01, rect.get_y()+rect.get_height()/2,
                f'{rect.get_width():.0f}', va='center', fontsize=9, color='#4b5563')
fig.tight_layout()
fig.savefig(os.path.join(ASSETS, 'chi2_polarity.png'), dpi=200, bbox_inches='tight', facecolor='white')
plt.close(fig); print('saved chi2_polarity.png')

json.dump(R, open(os.path.join(OUT, 'stats.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('DONE')
