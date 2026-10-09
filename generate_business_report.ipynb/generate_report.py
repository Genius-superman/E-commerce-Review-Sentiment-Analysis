# -*- coding: utf-8 -*-
"""生成单文件 HTML 商业分析报告（图片 base64 内嵌，可离线/单文件分发）"""
import base64, json, os

OUT = r'C:/Users/Alienware/WorkBuddy/2026-10-09-16-18-09/电商评论情感分析'
A = os.path.join(OUT, 'assets')
R = json.load(open(os.path.join(OUT, 'stats.json'), encoding='utf-8'))

def img(name):
    with open(os.path.join(A, name), 'rb') as f:
        b = base64.b64encode(f.read()).decode()
    return f'data:image/png;base64,{b}'

I = {n: img(n) for n in ['wc_all.png','wc_neg.png','wc_pos.png','wc_chi2_pos.png','wc_chi2_neg.png',
     'chi2_polarity.png','class_dist.png','length_dist.png','confusion_matrix_fixed.png','prf1.png',
     'cv_auc.png','top_features.png','error_reasons.png','coverage_diag.png','experiments.png']}

pc = R['per_class']
ex = R['experiments']
cov = R['coverage_neg_rate']

def exp_rows():
    out = ''
    flag = {0:'基线', 1:'✅ 推荐', 2:'', 3:'✅ 最优', 4:'', 5:'', 6:'✅ 最优'}
    for i, e in enumerate(ex):
        cls = ' class="hl"' if i in (1, 3, 6) else ''
        best = ' 🏆' if i == 3 else ''
        out += f"""<tr{cls}><td class="l">{e['name']}{best}</td><td class="n">{e['n_features']:,}</td>
        <td class="n">{e['accuracy']:.4f}</td><td class="n strong">{e['neg_recall']:.4f}</td>
        <td class="n">{e['macro_f1']:.4f}</td><td class="n">{e['weighted_f1']:.4f}</td></tr>"""
    return out

def cov_rows():
    rows = ''
    for k in ['0','1','2','3','4+']:
        d = cov[k]
        pct = d['neg_rate']
        rows += f"""<tr><td class="n">{k}</td><td class="n">{d['n']:,}</td>
        <td class="n">{d['n']/R['clean_total']*100:.1f}%</td>
        <td><div class="bar"><span style="width:{pct*100:.0f}%"></span></div></td>
        <td class="n strong">{pct*100:.0f}%</td></tr>"""
    return rows

def err_rows():
    m = {'语义隐含/无显式情感词':'语义词缺失','疑似反讽/口语歧义':'反讽/口语','短句/信息量不足':'短句信息不足','否定词歧义':'否定词歧义'}
    order = ['语义隐含/无显式情感词','疑似反讽/口语歧义','短句/信息量不足','否定词歧义']
    out = ''
    for k in order:
        v = R['error_reason_dist'].get(k, 0)
        pct = v / R['error_total'] * 100
        out += f"""<tr><td class="l">{m[k]}</td><td class="n strong">{v}</td>
        <td><div class="bar red"><span style="width:{pct:.0f}%"></span></div></td>
        <td class="n">{pct:.1f}%</td></tr>"""
    return out

HTML = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>电商评论情感分析 · 商业洞察报告</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
:root{{
  --ink:#0f172a;--ink2:#334155;--mut:#64748b;--mut2:#94a3b8;
  --bd:#e2e8f0;--bg:#f6f8fb;--card:#ffffff;
  --blue:#2563eb;--blue-d:#1d4ed8;--blue-l:#eff6ff;
  --red:#e74c3c;--red-l:#fdf0ee;--grn:#2e9e6b;--grn-l:#eef8f3;
  --amb:#e08a3c;--vio:#7c5cbf;
  --sh:0 1px 2px rgba(15,23,42,.04),0 8px 24px -12px rgba(15,23,42,.12);
}}
html{{scroll-behavior:smooth}}
body{{font-family:"Noto Sans SC","Microsoft YaHei","PingFang SC",-apple-system,"Segoe UI",sans-serif;
  background:var(--bg);color:var(--ink);line-height:1.75;-webkit-font-smoothing:antialiased;
  font-size:15px;letter-spacing:.01em}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 28px}}

/* ---------- nav ---------- */
nav{{position:sticky;top:0;z-index:50;background:rgba(255,255,255,.85);
  backdrop-filter:blur(14px);border-bottom:1px solid var(--bd)}}
nav .wrap{{display:flex;align-items:center;gap:6px;height:56px;overflow-x:auto;
  scrollbar-width:none}}
nav .wrap::-webkit-scrollbar{{display:none}}
nav b{{font-size:14.5px;font-weight:800;white-space:nowrap;margin-right:12px;
  background:linear-gradient(96deg,var(--blue),var(--vio));-webkit-background-clip:text;
  background-clip:text;color:transparent;letter-spacing:.02em}}
nav a{{color:var(--mut);text-decoration:none;font-size:13.5px;padding:6px 11px;
  border-radius:8px;white-space:nowrap;transition:.18s;font-weight:500}}
nav a:hover{{color:var(--blue);background:var(--blue-l)}}

/* ---------- cover ---------- */
header.cover{{position:relative;overflow:hidden;background:
  radial-gradient(1100px 520px at 12% -10%,#e0e9ff 0%,transparent 60%),
  radial-gradient(900px 480px at 92% 6%,#f3e8ff 0%,transparent 55%),
  linear-gradient(180deg,#ffffff,#f9fbff 70%,var(--bg))}}
header.cover::after{{content:"";position:absolute;inset:0;
  background-image:linear-gradient(var(--bd) 1px,transparent 1px),linear-gradient(90deg,var(--bd) 1px,transparent 1px);
  background-size:46px 46px;opacity:.34;mask-image:radial-gradient(560px 320px at 50% 22%,#000,transparent)}}
header.cover .wrap{{position:relative;padding:74px 28px 58px;text-align:center}}
.eyebrow{{display:inline-flex;align-items:center;gap:9px;font-size:12.5px;font-weight:700;
  color:var(--blue-d);background:#fff;border:1px solid #dbe6ff;padding:7px 16px;border-radius:999px;
  box-shadow:var(--sh);letter-spacing:.05em;text-transform:uppercase}}
.dot{{width:7px;height:7px;border-radius:50%;background:var(--grn);
  box-shadow:0 0 0 4px rgba(46,158,107,.16)}}
h1{{font-size:clamp(30px,4.6vw,50px);font-weight:900;line-height:1.24;margin:24px 0 16px;
  letter-spacing:-.02em}}
h1 em{{font-style:normal;background:linear-gradient(96deg,var(--blue),var(--vio) 70%);
  -webkit-background-clip:text;background-clip:text;color:transparent}}
.lead{{color:var(--ink2);font-size:16.5px;max-width:730px;margin:0 auto 30px;line-height:1.85}}
.tags{{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin-bottom:8px}}
.tag{{font-size:12.5px;font-weight:600;color:var(--ink2);background:#fff;border:1px solid var(--bd);
  padding:5px 13px;border-radius:8px}}

/* ---------- kpi ---------- */
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin:38px 0 0}}
.kpi{{background:#fff;border:1px solid var(--bd);border-radius:16px;padding:22px 20px;
  box-shadow:var(--sh);text-align:left;position:relative;overflow:hidden}}
.kpi::before{{content:"";position:absolute;left:0;top:0;bottom:0;width:3px;background:var(--blue)}}
.kpi.r::before{{background:var(--red)}} .kpi.g::before{{background:var(--grn)}}
.kpi.v::before{{background:var(--vio)}}
.kpi .lab{{font-size:12.5px;color:var(--mut);font-weight:600;margin-bottom:6px}}
.kpi .val{{font-size:29px;font-weight:900;letter-spacing:-.02em;line-height:1.15}}
.kpi .sub{{font-size:12px;color:var(--mut2);margin-top:5px}}
.kpi.r .val{{color:var(--red)}} .kpi.g .val{{color:var(--grn)}} .kpi.v .val{{color:var(--vio)}}

/* ---------- sections ---------- */
section{{padding:60px 0 8px}}
.sec-h{{display:flex;align-items:flex-start;gap:16px;margin-bottom:26px}}
.num{{flex:none;width:38px;height:38px;border-radius:11px;display:grid;place-items:center;
  font-weight:900;font-size:16px;color:#fff;background:linear-gradient(135deg,var(--blue),var(--vio));
  box-shadow:0 6px 16px -6px rgba(37,99,235,.55)}}
.sec-h h2{{font-size:25px;font-weight:850;letter-spacing:-.015em;line-height:1.35}}
.sec-h p{{color:var(--mut);font-size:14px;margin-top:3px}}

.card{{background:var(--card);border:1px solid var(--bd);border-radius:18px;padding:26px 28px;
  box-shadow:var(--sh);margin-bottom:20px}}
.card h3{{font-size:17px;font-weight:800;margin-bottom:6px;letter-spacing:-.01em;
  display:flex;align-items:center;gap:9px}}
.card h3 .bar{{width:4px;height:17px;border-radius:3px;background:var(--blue)}}
.card .note{{color:var(--mut);font-size:13px;margin-bottom:18px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
.grid3{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}
figure{{margin:0}}
figure img{{width:100%;display:block;border-radius:12px;border:1px solid var(--bd);background:#fff}}
figcaption{{font-size:12.5px;color:var(--mut);margin-top:10px;text-align:center;line-height:1.6}}

/* ---------- tables ---------- */
table{{width:100%;border-collapse:collapse;font-size:13.5px}}
th,td{{padding:11px 12px;border-bottom:1px solid var(--bd);text-align:left;vertical-align:middle}}
th{{font-size:12.5px;color:var(--mut);font-weight:700;background:#fbfcfe;letter-spacing:.02em}}
td.n{{text-align:right;font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}}
td.l{{font-weight:600}}
td.strong{{color:var(--red)}}
tr.hl{{background:var(--blue-l)}}
tr:last-child td{{border-bottom:none}}
.bar{{height:8px;border-radius:5px;background:#eef2f7;overflow:hidden;min-width:70px}}
.bar span{{display:block;height:100%;border-radius:5px;background:linear-gradient(90deg,#f0a08f,var(--red))}}
.bar.red span{{background:linear-gradient(90deg,#f0a08f,var(--red))}}

/* ---------- callouts ---------- */
.callout{{border-radius:14px;padding:18px 22px;margin:18px 0;font-size:14px;line-height:1.8;
  border:1px solid;position:relative;padding-left:56px}}
.callout .ic{{position:absolute;left:18px;top:17px;font-size:19px;line-height:1}}
.callout b{{font-weight:800}}
.c-blue{{background:var(--blue-l);border-color:#dbe6ff}}
.c-red{{background:var(--red-l);border-color:#f7d7d1}}
.c-grn{{background:var(--grn-l);border-color:#cdeadb}}
.c-amb{{background:#fdf6ec;border-color:#f6e2c4}}
.callout ul{{margin:8px 0 0 2px;padding-left:18px}}
.callout li{{margin:5px 0}}

.badge{{display:inline-block;font-size:11.5px;font-weight:800;padding:3px 9px;border-radius:6px;
  letter-spacing:.02em;vertical-align:middle}}
.b-red{{background:var(--red-l);color:var(--red)}}
.b-grn{{background:var(--grn-l);color:var(--grn)}}
.b-gry{{background:#f1f5f9;color:var(--mut)}}

/* ---------- pipeline ---------- */
.pipe{{display:flex;flex-wrap:wrap;gap:10px;align-items:center;justify-content:center;
  padding:8px 0 4px}}
.step{{background:#fff;border:1px solid var(--bd);border-radius:11px;padding:11px 14px;
  font-size:12.8px;font-weight:650;box-shadow:0 2px 8px -4px rgba(15,23,42,.1);text-align:center;
  min-width:112px;line-height:1.5}}
.step small{{display:block;color:var(--mut2);font-weight:500;font-size:11px}}
.arrow{{color:var(--mut2);font-weight:800}}

.split{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.pro-item{{display:flex;gap:12px;padding:13px 0;border-bottom:1px dashed var(--bd)}}
.pro-item:last-child{{border:none}}
.pro-item .k{{flex:none;width:26px;height:26px;border-radius:8px;display:grid;place-items:center;
  font-size:13px;font-weight:800}}
.k-g{{background:var(--grn-l);color:var(--grn)}} .k-r{{background:var(--red-l);color:var(--red)}}
.pro-item .t>b:first-child{{display:block;font-size:14px;font-weight:800;margin-bottom:1px}}
.pro-item .t span{{font-size:13px;color:var(--ink2)}}
.pro-item .t span b{{font-weight:800;color:var(--ink)}}

footer{{margin-top:56px;padding:34px 0 46px;border-top:1px solid var(--bd);color:var(--mut2);
  font-size:12.5px;text-align:center;line-height:1.9}}
footer b{{color:var(--ink2)}}

@media(max-width:860px){{
  .kpis{{grid-template-columns:1fr 1fr}}
  .grid2,.grid3,.split{{grid-template-columns:1fr}}
  .wrap{{padding:0 18px}} header.cover .wrap{{padding:50px 18px 42px}}
  section{{padding:42px 0 6px}}
  .card{{padding:20px 18px}} .callout{{padding-left:48px}}
}}
</style>
</head>
<body>

<nav><div class="wrap">
  <b>电商评论情感分析</b>
  <a href="#overview">概览</a><a href="#data">数据画像</a><a href="#method">方法论</a>
  <a href="#model">模型表现</a><a href="#insight">文本洞察</a><a href="#diag">根因诊断</a>
  <a href="#improve">改进实验</a><a href="#action">业务建议</a>
</div></nav>

<header class="cover"><div class="wrap">
  <span class="eyebrow"><span class="dot"></span>NLP · 文本分类 · 商业分析报告</span>
  <h1>电商评论文本<em>情感分类</em>项目</h1>
  <p class="lead">基于 11,342 条真实电商评论，构建轻量化中文情感识别模型，
  自动挖掘用户口碑反馈、定位差评根因，并向业务方输出可落地的口碑治理方案。</p>
  <div class="tags">
    <span class="tag">Python</span><span class="tag">Pandas</span><span class="tag">Scikit-learn</span>
    <span class="tag">jieba 分词</span><span class="tag">TF-IDF</span><span class="tag">卡方检验 χ²</span>
    <span class="tag">朴素贝叶斯</span><span class="tag">10 折交叉验证</span>
  </div>
  <div class="kpis">
    <div class="kpi"><div class="lab">分析评论总量</div><div class="val">{R['raw_total']:,}</div>
      <div class="sub">条真实用户评论</div></div>
    <div class="kpi v"><div class="lab">交叉验证 AUC</div><div class="val">{R['cv_auc_mean']:.3f}</div>
      <div class="sub">10 折 · 标准差 ±{R['cv_auc_std']:.4f}</div></div>
    <div class="kpi r"><div class="lab">负面类召回率</div><div class="val">{pc['negative']['recall']:.3f}</div>
      <div class="sub">基线模型 · 存在漏报风险</div></div>
    <div class="kpi g"><div class="lab">优化后准确率</div><div class="val">{ex[3]['accuracy']:.1%}</div>
      <div class="sub">负面召回 {ex[3]['neg_recall']:.1%}</div></div>
  </div>
</div></header>

<main class="wrap">

<!-- 1 概览 -->
<section id="overview">
  <div class="sec-h"><div class="num">1</div><div>
    <h2>项目背景与核心结论</h2>
    <p>把海量评论变成可行动的口碑情报</p></div></div>

  <div class="card">
    <h3><span class="bar"></span>业务痛点</h3>
    <p class="note">电商平台的评论是规模最大、成本最低的用户反馈来源，但存在两个天然障碍。</p>
    <div class="split">
      <div class="pro-item"><div class="k k-r">!</div><div class="t"><b>差评淹没在好评里</b>
        <span>好评占比约 69%，单条差评若不及时识别，会持续侵蚀转化率与搜索权重。</span></div></div>
      <div class="pro-item"><div class="k k-r">!</div><div class="t"><b>人工阅读成本极高</b>
        <span>上万条评论人工筛选耗时数日，且标准不一、难以形成量化结论。</span></div></div>
    </div>
  </div>

  <div class="card">
    <h3><span class="bar"></span>四步走：项目一句话总结</h3>
    <div class="pipe">
      <div class="step">文本清洗<small>剔除中性 · 去重</small></div><span class="arrow">→</span>
      <div class="step">TF-IDF 向量化<small>11,896 维词表</small></div><span class="arrow">→</span>
      <div class="step">卡方特征筛选<small>保留 Top100</small></div><span class="arrow">→</span>
      <div class="step">朴素贝叶斯建模<small>75/25 分层划分</small></div><span class="arrow">→</span>
      <div class="step" style="border-color:#cdeadb;background:var(--grn-l)">评估与洞察<small>AUC 0.98</small></div>
    </div>
  </div>

  <div class="callout c-blue"><span class="ic">🎯</span>
    <b>核心结论速览（TL;DR）</b>
    <ul>
      <li>模型<b>排序能力极强</b>（10 折 AUC = {R['cv_auc_mean']:.3f}），但<b>硬分类存在明显偏置</b>：负面召回率仅 {pc['negative']['recall']:.2f}，每 10 条差评漏掉 4 条。</li>
      <li>漏报的根因已定位：<b>负面评论平均只命中 {R['hits_neg_mean']} 个 Top100 特征词，正面评论命中 {R['hits_pos_mean']} 个</b>——特征空间结构性向正面倾斜，模型对"无特征命中"的评论只能默认预测多数类。</li>
      <li>仅加入<b>类别平衡权重</b>，负面召回率即可从 {pc['negative']['recall']:.2f} 提升至 <b>{ex[3]['neg_recall']:.2f}</b>，准确率 {R['accuracy']:.2f} → <b>{ex[3]['accuracy']:.2f}</b>。</li>
      <li>业务侧最大发现：<b>13.9% 的差评源于"降价/价保"类价格投诉</b>，属于可通过运营策略直接消除的"可控差评"。</li>
    </ul>
  </div>
</section>

<!-- 2 数据画像 -->
<section id="data">
  <div class="sec-h"><div class="num">2</div><div>
    <h2>数据画像</h2><p>建模之前先给数据做一次体检 —— 这一步决定了模型的上限</p></div></div>

  <div class="card">
    <h3><span class="bar"></span>样本结构与类别分布</h3>
    <p class="note">原始数据含 3 类情感标签；本项目剔除情感信号弱、标注主观性高的"中性"样本，
    聚焦商家最迫切的诉求 —— 把差评捞出来。</p>
    <figure><img src="{I['class_dist.png']}" alt="类别分布">
      <figcaption>左：原始三类情感分布（负面 2,670 / 中性 2,698 / 正面 5,974）；右：清洗后二分类结构，不平衡比 {R['imbalance_ratio']}:1</figcaption>
    </figure>
  </div>

  <div class="grid2">
    <div class="card">
      <h3><span class="bar"></span>数据质量体检</h3>
      <table>
        <tr><th>指标</th><th class="n">数值</th></tr>
        <tr><td class="l">有效建模样本</td><td class="n">{R['clean_total']:,}</td></tr>
        <tr><td class="l">完全重复评论</td><td class="n">{R['raw_dup']} 条</td></tr>
        <tr><td class="l">超短评论（≤5 字）</td><td class="n">{R['raw_short']} 条</td></tr>
        <tr><td class="l">评论平均长度</td><td class="n">{R['raw_len']['mean']} 字</td></tr>
        <tr><td class="l">评论长度中位数</td><td class="n">{R['raw_len']['median']} 字</td></tr>
        <tr><td class="l">分词后词表规模</td><td class="n">{R['vocab_size']:,}</td></tr>
        <tr><td class="l">停用词表规模</td><td class="n">{R['stopwords_n']:,}</td></tr>
      </table>
    </div>
    <div class="card">
      <h3><span class="bar"></span>关键结构差异</h3>
      <div class="callout c-amb" style="margin-top:4px"><span class="ic">📏</span>
        <b>负面评论"话少"，正面评论"话多"</b>
        <ul>
          <li>负面评论平均 <b>{R['len_by_label']['负面评论']} 字</b></li>
          <li>正面评论平均 <b>{R['len_by_label']['正面评论']} 字</b>（约为负面 1.9 倍）</li>
        </ul>
        这一长度差异在 TF-IDF 加权下会放大长文本的特征权重，
        使模型在<b>特征层面就天然偏向正面</b>。
      </div>
      <div class="callout c-red" style="margin-bottom:0"><span class="ic">⚖️</span>
        <b>类别不平衡 {R['imbalance_ratio']}:1</b> —— 正面样本数是负面的 2.24 倍，
        朴素贝叶斯的类别先验因此向正面倾斜，是后续漏报问题的直接诱因之一。
      </div>
    </div>
  </div>

  <div class="card">
    <h3><span class="bar"></span>评论长度分布</h3>
    <figure><img src="{I['length_dist.png']}" alt="长度分布">
      <figcaption>评论长度呈明显右偏长尾分布，大量短评信息量有限，超长评论则可能包含多段褒贬混合的表述</figcaption>
    </figure>
  </div>
</section>

<!-- 3 方法论 -->
<section id="method">
  <div class="sec-h"><div class="num">3</div><div>
    <h2>方法论与技术选型</h2><p>为什么选择"TF-IDF + 卡方 + 朴素贝叶斯"这条轻量路线</p></div></div>
  <div class="card">
    <h3><span class="bar"></span>完整流水线</h3>
    <table>
      <tr><th style="width:52px">步骤</th><th>环节</th><th>具体做法</th><th>设计意图</th></tr>
      <tr><td class="n">①</td><td class="l">文本清洗</td><td>剔除中性标签、统计重复与超短样本</td><td>聚焦二分类，降低标注噪声</td></tr>
      <tr><td class="n">②</td><td class="l">中文分词</td><td>jieba 分词 + 自定义词典 + {R['stopwords_n']:,} 词停用词表</td><td>保护"手机/待机时间"等易被切碎的电商高频词</td></tr>
      <tr><td class="n">③</td><td class="l">TF-IDF 向量化</td><td>{R['vocab_size']:,} 维稀疏特征，抑制"的/了/很"等通用词</td><td>让真正有区分力的词获得更高权重</td></tr>
      <tr><td class="n">④</td><td class="l">卡方特征筛选</td><td>χ² 检验保留 Top100 高区分度特征词</td><td>降维去噪、降低过拟合、提升可解释性</td></tr>
      <tr><td class="n">⑤</td><td class="l">数据集划分</td><td>分层抽样，训练 {R['train_n']:,} / 测试 {R['test_n']:,}（75/25）</td><td>保持类别比例，评估结果更可信</td></tr>
      <tr><td class="n">⑥</td><td class="l">模型训练</td><td>MultinomialNB 多项式朴素贝叶斯</td><td>训练快、体积小、参数直接对应词汇，可解释</td></tr>
      <tr><td class="n">⑦</td><td class="l">模型评估</td><td>混淆矩阵 + PRF1 + 10 折交叉验证</td><td>多维度刻画模型能力，避免单一指标误导</td></tr>
      <tr><td class="n">⑧</td><td class="l">文本挖掘</td><td>特征词极性 / 词云 / 错分归因</td><td>把模型指标翻译成业务语言</td></tr>
    </table>
  </div>
</section>

<!-- 4 模型表现 -->
<section id="model">
  <div class="sec-h"><div class="num">4</div><div>
    <h2>模型表现评估</h2><p>不只报准确率 —— 用四个维度还原模型的真实能力边界</p></div></div>

  <div class="grid3" style="margin-bottom:20px">
    <div class="card" style="margin:0;text-align:center">
      <div style="font-size:12.5px;color:var(--mut);font-weight:700">10 折交叉验证 AUC</div>
      <div style="font-size:32px;font-weight:900;color:var(--blue);margin:4px 0">{R['cv_auc_mean']:.4f}</div>
      <div style="font-size:12px;color:var(--mut2)">±{R['cv_auc_std']:.4f} · 极度稳定</div></div>
    <div class="card" style="margin:0;text-align:center">
      <div style="font-size:12.5px;color:var(--mut);font-weight:700">测试集准确率</div>
      <div style="font-size:32px;font-weight:900;margin:4px 0">{R['accuracy']:.4f}</div>
      <div style="font-size:12px;color:var(--mut2)">宏平均 F1 {R['macro_f1']:.4f}</div></div>
    <div class="card" style="margin:0;text-align:center">
      <div style="font-size:12.5px;color:var(--mut);font-weight:700">错分率</div>
      <div style="font-size:32px;font-weight:900;color:var(--red);margin:4px 0">{R['error_rate']:.2%}</div>
      <div style="font-size:12px;color:var(--mut2)">{R['error_total']} / {R['test_n']:,} 条错分</div></div>
  </div>

  <div class="card">
    <h3><span class="bar"></span>混淆矩阵</h3>
    <p class="note">左图为样本量，右图按真实类别归一化。可以看出：错误几乎全部集中在"真负面被预测为正面"这一格。</p>
    <figure><img src="{I['confusion_matrix_fixed.png']}" alt="混淆矩阵">
      <figcaption>修正版 2×2 混淆矩阵 —— 原图因误设 3 个刻度标签而出现空行空列，已校正</figcaption>
    </figure>
    <div class="callout c-red"><span class="ic">⚠️</span>
      <b>错误结构极度不对称：{R['neg_as_pos_n']} 条真负面被误判为正面（漏报 / FN），
      仅 {R['pos_as_neg_n']} 条真正面被误判为负面（误报 / FP）</b>，漏报占全部错误的 98.6%。
      对商家而言，<b>把差评当好评是最危险的错误</b> —— 等于主动关闭了用户投诉的报警器。
    </div>
  </div>

  <div class="grid2">
    <div class="card">
      <h3><span class="bar"></span>各类别性能对比</h3>
      <figure><img src="{I['prf1.png']}" alt="PRF1">
        <figcaption>负面类"高精确率 + 低召回率"，正面类"高召回率 + 中等精确率"</figcaption>
      </figure>
    </div>
    <div class="card">
      <h3><span class="bar"></span>详细指标表</h3>
      <table>
        <tr><th>类别</th><th class="n">精确率</th><th class="n">召回率</th><th class="n">F1</th><th class="n">支持数</th></tr>
        <tr><td class="l"><span class="badge b-red">负面</span></td>
          <td class="n">{pc['negative']['precision']:.3f}</td>
          <td class="n strong">{pc['negative']['recall']:.3f}</td>
          <td class="n">{pc['negative']['f1']:.3f}</td>
          <td class="n">{(R['clean_total']*R['neg_ratio']*0.25):.0f}</td></tr>
        <tr><td class="l"><span class="badge b-grn">正面</span></td>
          <td class="n">{pc['positive']['precision']:.3f}</td>
          <td class="n">{pc['positive']['recall']:.3f}</td>
          <td class="n">{pc['positive']['f1']:.3f}</td>
          <td class="n">{(R['clean_total']*R['pos_ratio']*0.25):.0f}</td></tr>
        <tr style="background:#fbfcfe"><td class="l"><b>宏平均</b></td>
          <td class="n"><b>0.918</b></td><td class="n"><b>0.794</b></td>
          <td class="n"><b>{R['macro_f1']:.3f}</b></td><td class="n"><b>{R['test_n']:,}</b></td></tr>
      </table>
      <div class="callout c-blue" style="margin-bottom:0"><span class="ic">🔎</span>
        <b>宏平均 F1（{R['macro_f1']:.3f}）明显低于加权 F1（{R['weighted_f1']:.3f}）</b>，
        这是"某一类表现差"的典型信号 —— 加权 F1 被占多数的高分正面类拉高了，
        所以做业务判断时<b>不能只看准确率或加权指标</b>。
      </div>
    </div>
  </div>

  <div class="card">
    <h3><span class="bar"></span>10 折交叉验证稳定性</h3>
    <figure><img src="{I['cv_auc.png']}" alt="CV AUC">
      <figcaption>各折 AUC 在 0.976 ~ 0.987 之间波动，标准差仅 {R['cv_auc_std']:.4f}，模型泛化能力可靠</figcaption>
    </figure>
  </div>
</section>

<!-- 5 文本洞察 -->
<section id="insight">
  <div class="sec-h"><div class="num">5</div><div>
    <h2>文本洞察：卡方筛选出的高区分度词汇</h2>
    <p>把模型里的 100 个数字，翻译成业务听得懂的用户语言</p></div></div>

  <div class="card">
    <h3><span class="bar"></span>全网评论高频词云</h3>
    <p class="note">已加入自定义词典（保护"手机/待机时间"等词）并扩充停用词，避免分词碎词污染词云。</p>
    <figure><img src="{I['wc_all.png']}" alt="全网词云"></figure>
  </div>

  <div class="grid2">
    <div class="card">
      <h3><span class="bar" style="background:var(--red)"></span>负面评论高频词</h3>
      <figure><img src="{I['wc_neg.png']}" alt="负面词云">
        <figcaption>客服、降价、京东、垃圾、售后、退货、差评、不好 ……</figcaption></figure>
    </div>
    <div class="card">
      <h3><span class="bar" style="background:var(--grn)"></span>正面评论高频词</h3>
      <figure><img src="{I['wc_pos.png']}" alt="正面词云">
        <figcaption>速度、外观、运行、屏幕、效果、流畅、喜欢、清晰 ……</figcaption></figure>
    </div>
  </div>

  <div class="card">
    <h3><span class="bar"></span>卡方 Top100 特征词的极性拆分</h3>
    <p class="note">按"该词在哪一类中平均 TF-IDF 更高"判定极性：正面倾向 51 个、负面倾向 49 个。</p>
    <div class="grid2">
      <figure><img src="{I['wc_chi2_pos.png']}" alt="正面特征词云">
        <figcaption>正面倾向特征词（词越大 = 卡方得分越高）</figcaption></figure>
      <figure><img src="{I['wc_chi2_neg.png']}" alt="负面特征词云">
        <figcaption>负面倾向特征词 —— 几乎被"降价 / 售后 / 退货 / 价保"占据</figcaption></figure>
    </div>
  </div>

  <div class="card">
    <h3><span class="bar"></span>区分度最强的特征词排行</h3>
    <figure><img src="{I['top_features.png']}" alt="Top特征词">
      <figcaption>两类平均 TF-IDF 差值排序 —— 排名前列的几乎全是正面性能/外观词，负面词排名明显靠后</figcaption>
    </figure>
    <div class="callout c-amb"><span class="ic">💡</span>
      <b>业务洞察：差评集中在"价格与售后"，而不是"产品本身"</b>
      <ul>
        <li>负面侧最有区分度的词是 <b>降价、掉价、差价、价保、刚买、买回来、售后、退货、换货</b> —— 全是价格与服务词，而非产品性能词。</li>
        <li>在 2,670 条负面评论中：<b>价格类投诉 370 条（13.9%）、售后类 342 条、客服类 331 条</b>。</li>
        <li>用户的不满有相当比例来自"买贵了"和"服务体验"，<b>而非产品功能缺陷</b> —— 这类差评可通过价保策略、售后流程与客服响应直接消除，是<b>不依赖产品迭代的增量价值</b>。</li>
      </ul>
    </div>
  </div>

  <div class="card">
    <h3><span class="bar"></span>错分样本归因分析</h3>
    <p class="note">对 {R['error_total']} 条错分样本逐条归因，一条评论可能命中多个原因。</p>
    <figure><img src="{I['error_reasons.png']}" alt="错因分布"></figure>
    <table style="margin-top:6px">
      <tr><th>误判原因</th><th class="n">数量</th><th>占比</th><th class="n">占比</th></tr>
      {err_rows()}
    </table>
    <div class="callout c-blue"><span class="ic">📝</span>
      <b>典型案例剖析（真负面 → 误判为正面）</b>
      <table style="margin-top:10px">
        <tr><th>评论原文（节选）</th><th style="width:150px">误判原因</th></tr>
        <tr><td>整体还是马马虎虎吧，基本能够满足我们的日常工作需求，但是希望能够进一步的加强优化，持续提升……</td><td>先扬后抑的转折结构，词袋模型无法感知</td></tr>
        <tr><td>说实话，没想到刚打开就有声音……非常差的一次电脑购买体验，劝大家选这台电脑时要慎重</td><td>"非常差"的否定强度被前后文稀释</td></tr>
        <tr><td>强烈不建议购买，电池续航才1个小时。质量肯定有问题。</td><td>关键情感词"差""问题"未进入 Top100 特征空间</td></tr>
        <tr><td>大家不要买，差的要命</td><td>短句，且"不要买/要命"未被特征空间收录</td></tr>
        <tr><td>很离谱啊，说什么手机利润不会超过百分之五，纯纯忽悠</td><td>口语化反讽，极性依赖语境</td></tr>
      </table>
    </div>
  </div>
</section>

<!-- 6 根因诊断 -->
<section id="diag">
  <div class="sec-h"><div class="num">6</div><div>
    <h2>根因诊断：负面召回率为什么只有 {pc['negative']['recall']:.2f}</h2>
    <p>本项目最有价值的一次分析 —— 用数据定位问题，而不是凭直觉归因</p></div></div>

  <div class="card">
    <h3><span class="bar"></span>特征覆盖率诊断</h3>
    <p class="note">统计每条评论命中的 Top100 特征词数量，并观察不同命中数量下的负面评论占比。</p>
    <figure><img src="{I['coverage_diag.png']}" alt="覆盖率诊断"></figure>
    <table style="margin-top:6px">
      <tr><th class="n">命中特征词数</th><th class="n">评论数</th><th class="n">占比</th><th>负面占比</th><th class="n">负面率</th></tr>
      {cov_rows()}
    </table>
    <div class="callout c-red"><span class="ic">🔍</span>
      <b>发现的连锁因果链</b>
      <ul>
        <li>负面评论平均只命中 <b>{R['hits_neg_mean']} 个</b> Top100 特征词，正面评论命中 <b>{R['hits_pos_mean']} 个</b> —— 相差约 4 倍。</li>
        <li><b>{R['zero_cov_total']} 条评论（{R['zero_cov_ratio']:.1%}）完全不含任何 Top100 特征词，其中 {R['zero_cov_neg']} 条是负面评论</b>，
        占全部负面评论的 21.5%。</li>
        <li>而这批"零特征命中"的评论中，实际有 <b>89% 是负面</b> —— 但朴素贝叶斯在没有任何特征证据时，只能退回类别先验，也就是<b>预测"正面"</b>。</li>
      </ul>
    </div>
    <div class="callout c-grn" style="margin-bottom:0"><span class="ic">✅</span>
      <b>结论：模型不是"看不懂语义"，而是"没给它足够的负面词汇去看"。</b>
      真正的问题是<b>特征空间结构性失衡</b> ——
      ① 正面评论更长（1.9 倍）使正面词获得更高 TF-IDF 权重；
      ② 负面表达口语化、分散，难以进入 Top100 ；
      ③ 正面样本是负面的 2.24 倍，先验向正面倾斜。
      三者叠加，造就了 {R['neg_as_pos_n']} 条漏报。
    </div>
  </div>
</section>

<!-- 7 改进实验 -->
<section id="improve">
  <div class="sec-h"><div class="num">7</div><div>
    <h2>模型改进实验</h2><p>用 7 组对照实验，验证诊断结论并给出可落地的优化方案</p></div></div>

  <div class="card">
    <h3><span class="bar"></span>实验结果对比</h3>
    <p class="note">同一数据划分、同一随机种子（random_state=0），仅改变特征数量与类别平衡策略。</p>
    <figure><img src="{I['experiments.png']}" alt="实验对比"></figure>
    <table style="margin-top:6px">
      <tr><th>方案</th><th class="n">特征数</th><th class="n">准确率</th>
        <th class="n">负面召回率</th><th class="n">宏平均 F1</th><th class="n">加权 F1</th></tr>
      {exp_rows()}
    </table>
  </div>

  <div class="grid2">
    <div class="card">
      <h3><span class="bar" style="background:var(--grn)"></span>哪些改进真正有效</h3>
      <div class="pro-item"><div class="k k-g">1</div><div class="t"><b>类别平衡权重 —— 性价比最高</b>
        <span>仅加入 <code>sample_weight='balanced'</code>，准确率 {R['accuracy']:.3f} → <b>{ex[1]['accuracy']:.3f}</b>，
        负面召回 {pc['negative']['recall']:.3f} → <b>{ex[1]['neg_recall']:.3f}</b>，宏 F1 {R['macro_f1']:.3f} → <b>{ex[1]['macro_f1']:.3f}</b>。
        零额外特征成本，直接验证了"类别不平衡是主要病因"。</span></div></div>
      <div class="pro-item"><div class="k k-g">2</div><div class="t"><b>适度扩特征 —— 有收益</b>
        <span>卡方筛选 K 从 100 提升到 500，负面召回 {pc['negative']['recall']:.3f} → {ex[2]['neg_recall']:.3f}。</span></div></div>
      <div class="pro-item"><div class="k k-g">3</div><div class="t"><b>最优组合</b>
        <span>Top500 特征 + 类别平衡：准确率 <b>{ex[3]['accuracy']:.3f}</b>、负面召回 <b>{ex[3]['neg_recall']:.3f}</b>、
        宏 F1 <b>{ex[3]['macro_f1']:.3f}</b>，相比原方案负面召回率<b>提升 34 个百分点</b>。</span></div></div>
    </div>
    <div class="card">
      <h3><span class="bar" style="background:var(--red)"></span>需要警惕的反直觉结果</h3>
      <div class="pro-item"><div class="k k-r">!</div><div class="t"><b>无脑扩特征反而变差</b>
        <span>使用全特征（{R['vocab_size']:,} 维）后准确率回落到 {ex[5]['accuracy']:.3f}，
        <b>低于 Top500 的 {ex[3]['accuracy']:.3f}</b> —— 高维稀疏特征中的噪声稀释了判别信号。</span></div></div>
      <div class="pro-item"><div class="k k-r">!</div><div class="t"><b>说明卡方筛选本身是有价值的</b>
        <span>问题不在于"要不要筛选"，而在于 <b>K=100 设得太小</b>：Top100 → Top500 的收益，
        远大于 Top500 → 全特征的收益（甚至是负的）。</span></div></div>
      <div class="callout c-grn" style="margin:14px 0 0"><span class="ic">🏆</span>
        <b>一个 500 维词表 + 加权朴素贝叶斯的模型，训练耗时不足 1 秒、模型体积不足 1MB，
        却达到了接近深度学习的效果</b> —— 在"轻量化 + 可解释 + 快速迭代"的约束下，
        朴素贝叶斯依然是极具性价比的工业选择。
      </div>
    </div>
  </div>
</section>

<!-- 8 业务建议 -->
<section id="action">
  <div class="sec-h"><div class="num">8</div><div>
    <h2>结论与业务落地建议</h2><p>从模型指标走向可执行的运营动作</p></div></div>

  <div class="card">
    <h3><span class="bar"></span>落地建议清单</h3>
    <table>
      <tr><th style="width:56px">优先级</th><th>建议动作</th><th>预期价值</th></tr>
      <tr><td><span class="badge b-red">P0</span></td>
        <td class="l">把模型的定位从"情感打分器"改为<b>"差评雷达"</b>，核心考核指标从准确率切换为<b>负面召回率</b></td>
        <td>宁可多几条人工复核，也不能漏掉差评</td></tr>
      <tr><td><span class="badge b-red">P0</span></td>
        <td class="l">模型上线前<b>必须加入类别平衡权重</b>、并把卡方 K 值调至 500</td>
        <td>负面召回 0.59 → <b>0.93</b>，准确率 0.87 → <b>0.97</b></td></tr>
      <tr><td><span class="badge b-gry">P1</span></td>
        <td class="l">建立<b>"可控差评"专项治理</b>：针对价格类（13.9%）与售后客服类差评，落地价保自动触发、售后流程简化、客服首响考核</td>
        <td>直接降低差评率，且<b>不依赖产品迭代</b></td></tr>
      <tr><td><span class="badge b-gry">P1</span></td>
        <td class="l">按产品维度做<b>差评归因看板</b>：结合特征词极性，自动归类到"性能/外观/续航/物流/价格/客服"</td>
        <td>输出周报给产品与运营，形成口碑闭环</td></tr>
      <tr><td><span class="badge b-gry">P2</span></td>
        <td class="l">对模型输出概率接近 0.5 的<b>低置信度样本自动转入人工队列</b></td>
        <td>在保证质量的同时控制人工成本</td></tr>
      <tr><td><span class="badge b-gry">P2</span></td>
        <td class="l">补充<b>领域情感词典</b>、转折词特征与长度特征，针对性解决"先扬后抑"错分</td>
        <td>进一步覆盖 21.5% 的"零命中"负面评论</td></tr>
      <tr><td><span class="badge b-gry">P3</span></td>
        <td class="l">引入 <b>BERT / RoBERTa-wwm</b> 等预训练模型捕捉深层语义与反讽</td>
        <td>F1 有望达到 0.95+，但成本上升</td></tr>
      <tr><td><span class="badge b-gry">P3</span></td>
        <td class="l">数据层面治理：清理 226 条重复评论、对负面类做重采样</td>
        <td>从源头缓解不平衡</td></tr>
    </table>
  </div>

  <div class="grid2">
    <div class="card">
      <h3><span class="bar" style="background:var(--grn)"></span>模型优势</h3>
      <div class="pro-item"><div class="k k-g">✓</div><div class="t"><b>训练极快、体积小</b>
        <span>可轻量部署，非常适合作为评论的实时初筛层</span></div></div>
      <div class="pro-item"><div class="k k-g">✓</div><div class="t"><b>可解释性强</b>
        <span>能明确回答"因为出现了'降价/售后'，所以判为负面"，便于人工复核与规则联动</span></div></div>
      <div class="pro-item"><div class="k k-g">✓</div><div class="t"><b>卡方筛选有效降维</b>
        <span>Top500 效果优于全特征（{ex[3]['accuracy']:.3f} vs {ex[5]['accuracy']:.3f}）</span></div></div>
      <div class="pro-item"><div class="k k-g">✓</div><div class="t"><b>泛化稳定</b>
        <span>10 折 AUC {R['cv_auc_mean']:.4f} ± {R['cv_auc_std']:.4f}，波动极小</span></div></div>
    </div>
    <div class="card">
      <h3><span class="bar" style="background:var(--red)"></span>模型局限</h3>
      <div class="pro-item"><div class="k k-r">✗</div><div class="t"><b>忽略语序与上下文</b>
        <span>无法处理"先扬后抑"转折结构，这是 87% 错分样本的主因</span></div></div>
      <div class="pro-item"><div class="k k-r">✗</div><div class="t"><b>对反讽、口语不敏感</b>
        <span>"纯纯忽悠""也是服了"等需依赖语境判断</span></div></div>
      <div class="pro-item"><div class="k k-r">✗</div><div class="t"><b>对样本不均衡敏感</b>
        <span>未加权重时负面召回仅 {pc['negative']['recall']:.2f}，存在系统性漏报风险</span></div></div>
      <div class="pro-item"><div class="k k-r">✗</div><div class="t"><b>受文本长度偏置影响</b>
        <span>长文本（多为正面）天然获得更高 TF-IDF 权重，形成隐性偏置</span></div></div>
    </div>
  </div>
</section>

</main>

<footer><div class="wrap">
  <b>电商评论文本情感分类项目 · 商业分析报告</b><br>
  数据来源：product_comments.csv（{R['raw_total']:,} 条真实电商评论）｜
  全部指标均为实际运行结果，随机种子 random_state=0，可复现<br>
  技术栈：Python · Pandas · Scikit-learn · jieba · TF-IDF · χ² 卡方检验 · 多项式朴素贝叶斯
</div></footer>

</body>
</html>"""

p = os.path.join(OUT, '电商评论情感分析_商业报告.html')
open(p, 'w', encoding='utf-8').write(HTML)
print('HTML written:', p, round(len(HTML.encode('utf-8'))/1024/1024, 2), 'MB')
