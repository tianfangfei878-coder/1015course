"""《全唐詩》LDA 主題模型（含文言停用詞過濾）
- 每首詩 = 一篇文件；以「字」為單位（不分詞），只保留漢字
- 停用詞：qhchina.helpers.texts.load_stopwords('zh_cl_sim')
- 訓練 10 與 20 個主題；每主題列前 10 字 + 2 首最典型詩
- 輸出：lda_results_sw.csv ＆ lda_results_sw.html
"""
import json, re, time, csv, warnings, html
import numpy as np
from qhchina.analytics.topicmodels import LDAGibbsSampler
from qhchina.helpers.texts import load_stopwords

SEED = 42
ITERATIONS = 100
N_WORDS, N_DOCS = 10, 2
CSV_PATH, HTML_PATH = 'lda_results_sw.csv', 'lda_results_sw.html'

# ---------- 載入語料 ----------
with open('全唐诗.json') as f:
    poems = json.load(f)

ch = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
docs, meta = [], []
for p in poems:
    text = ''.join(p['paragraphs'])
    toks = ch.findall(text)
    if toks:
        docs.append(toks)
        meta.append((p['title'], p['author'], text))

stopwords = load_stopwords('zh_cl_sim')
print(f'文件數: {len(docs)}  總字數: {sum(len(d) for d in docs)}  停用詞: {len(stopwords)}', flush=True)

rows, html_blocks = [], []
for k in (10, 20):
    t0 = time.time()
    lda = LDAGibbsSampler(n_topics=k, iterations=ITERATIONS,
                          random_state=SEED, log_interval=25,
                          stopwords=stopwords)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda.fit(docs)
    perp, elapsed = lda.perplexity(), time.time() - t0
    print(f'--- k={k}: vocab={lda.vocabulary_size}, {elapsed:.0f}s, perp={perp:.2f}', flush=True)

    topics = lda.get_topics(n_words=N_WORDS)
    block = [f'<h2>{k} 個主題</h2>',
             f'<p class="meta">文件數 {len(docs)} ｜ 詞表 {lda.vocabulary_size} 字（去除 {len(stopwords)} 個文言停用詞）'
             f'｜ 迭代 {ITERATIONS} 次 ｜ seed {SEED} ｜ perplexity {perp:.2f} ｜ 訓練 {elapsed:.0f} 秒</p>']
    for tid, tw in enumerate(topics):
        top10 = '、'.join(w for w, _ in tw)
        row = {'n_topics': k, 'topic_id': tid,
               'top10_words': top10,
               'top10_words_prob': ' '.join(f'{w}:{p:.4f}' for w, p in tw)}
        poem_cells = []
        for j, (doc_id, prob) in enumerate(lda.get_top_documents(tid, N_DOCS), 1):
            title, author, text = meta[doc_id]
            author = author or '（佚名）'
            row[f'poem{j}_title'] = title
            row[f'poem{j}_author'] = author
            row[f'poem{j}_theta'] = round(prob, 4)
            row[f'poem{j}_text'] = text
            poem_cells.append(
                f'<div class="poem"><div class="ptitle">{j}. {html.escape(title)} ／ {html.escape(author)}'
                f'<span class="theta">θ={prob:.3f}</span></div>'
                f'<div class="ptext">{html.escape(text)}</div></div>')
        wmax = tw[0][1]
        words_html = ''.join(
            f'<span class="w" style="font-size:{12 + 22 * p / wmax:.0f}px">{html.escape(w)}</span>'
            for w, p in tw)
        block.append(f'<div class="topic"><div class="tid">主題 {tid}</div>'
                     f'<div class="words">{words_html}</div>{"".join(poem_cells)}</div>')
        rows.append(row)
    html_blocks.append('\n'.join(block))

# ---------- CSV（utf-8-sig，Excel 開啟中文不亂碼） ----------
fields = ['n_topics', 'topic_id', 'top10_words', 'top10_words_prob',
          'poem1_title', 'poem1_author', 'poem1_theta', 'poem1_text',
          'poem2_title', 'poem2_author', 'poem2_theta', 'poem2_text']
with open(CSV_PATH, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    w.writerows(rows)

# ---------- HTML ----------
css = """body{font-family:"PingFang TC","Microsoft JhengHei",serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
h1{font-size:26px}h2{border-bottom:2px solid #8b0000;padding-bottom:4px;margin-top:40px}
.meta{color:#666;font-size:13px}
.topic{display:flex;gap:14px;margin:14px 0;padding:12px;background:#faf7f2;border:1px solid #e5ddcf;border-radius:8px;align-items:flex-start}
.tid{min-width:74px;font-weight:bold;color:#8b0000;padding-top:6px}
.words{min-width:280px;line-height:2}
.w{display:inline-block;margin-right:10px;color:#333}
.poem{flex:1;background:#fff;border-left:3px solid #c9a86a;padding:6px 10px;margin:4px 0;font-size:14px}
.ptitle{font-weight:600;font-size:13px}.theta{color:#888;font-weight:normal;font-size:12px;margin-left:8px}
.ptext{margin-top:4px;line-height:1.7}"""
html_doc = ('<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">'
            '<title>《全唐詩》LDA 主題模型</title><style>' + css + '</style></head><body>'
            '<h1>《全唐詩》主題模型（LDA Gibbs Sampling）</h1>'
            '<p class="meta">每首詩為一篇文件，以字為單位（不分詞、去除標點），'
            '並以 qhchina <code>zh_cl_sim</code> 文言停用詞表過濾虛詞。</p>'
            + '\n'.join(html_blocks) + '</body></html>')
with open(HTML_PATH, 'w', encoding='utf-8') as f:
    f.write(html_doc)

print(f'輸出：{CSV_PATH}（{len(rows)} 列）、{HTML_PATH}', flush=True)
