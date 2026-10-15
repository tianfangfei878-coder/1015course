"""《全唐詩》LDA 主題模型訓練
- 每首詩 = 一篇文件
- 以「字」為單位（不分詞），只保留漢字（即去掉標點符號與空白）
- 分別訓練 10 個與 20 個主題
- 每個主題列出前 10 個字，並舉 2 首最典型的詩
"""
import json, re, time, warnings
import numpy as np
from qhchina.analytics.topicmodels import LDAGibbsSampler

SEED = 42
ITERATIONS = 100
N_WORDS = 10
N_DOCS = 2

# ---------- 載入語料 ----------
with open('全唐诗.json') as f:
    poems = json.load(f)

ch = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')   # 只保留漢字 → 去掉標點
docs, meta = [], []
for p in poems:
    text = ''.join(p['paragraphs'])
    toks = ch.findall(text)
    if toks:
        docs.append(toks)
        meta.append((p['title'], p['author'], text))

print(f'文件數: {len(docs)}  總字數: {sum(len(d) for d in docs)}', flush=True)

out_lines = []
def log(s=''):
    print(s, flush=True)
    out_lines.append(s)

# ---------- 訓練 ----------
for k in (10, 20):
    log(f'\n{"="*70}\n【 {k} 個主題 】 iterations={ITERATIONS}, seed={SEED}\n{"="*70}')
    t0 = time.time()
    lda = LDAGibbsSampler(n_topics=k, iterations=ITERATIONS,
                          random_state=SEED, log_interval=25)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')   # 短詩（<24字）警告不影響訓練
        lda.fit(docs)
    log(f'訓練時間: {time.time()-t0:.1f}s | perplexity: {lda.perplexity():.2f}')

    topics = lda.get_topics(n_words=N_WORDS)
    for tid, tw in enumerate(topics):
        words = '  '.join(f'{w}({p:.4f})' for w, p in tw)
        log(f'\n主題 {tid:>2}: {words}')
        for j, (doc_id, prob) in enumerate(lda.get_top_documents(tid, N_DOCS), 1):
            title, author, text = meta[doc_id]
            log(f'   典型詩{j} (θ={prob:.3f}): {title} ／ {author}')
            log(f'      {text}')

    # 儲存核心結果（避免 npy 全模型快照佔用數 GB）
    np.savez_compressed(f'lda_k{k}.npz',
                        phi=lda.phi, theta=lda.theta,
                        vocab=np.array(lda.vocabulary, dtype=object),
                        n_dt=lda.n_dt, doc_lengths=lda.doc_lengths,
                        alpha=lda.alpha, beta=np.array(lda.beta))

with open('lda_results.txt', 'w') as f:
    f.write('\n'.join(out_lines))
print('\n結果已存到 lda_results.txt', flush=True)
