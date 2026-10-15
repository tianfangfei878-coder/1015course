"""計算各主題在語料中的實際占比（與 train_lda_stopwords.py 相同設定，seed 固定 → 結果可重現）
- token share：全語料字數中，被指派到該主題的比例（n_t / 總字數）
- doc dominant share：以該主題為「最主要主題」的詩的篇數比例
"""
import json, re, time, warnings
import numpy as np
from qhchina.analytics.topicmodels import LDAGibbsSampler
from qhchina.helpers.texts import load_stopwords

SEED, ITERATIONS = 42, 100
with open('全唐诗.json') as f:
    poems = json.load(f)
ch = re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff]')
docs = [ch.findall(''.join(p['paragraphs'])) for p in poems]
docs = [d for d in docs if d]
stopwords = load_stopwords('zh_cl_sim')

for k in (10, 20):
    lda = LDAGibbsSampler(n_topics=k, iterations=ITERATIONS, random_state=SEED,
                          stopwords=stopwords)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        lda.fit(docs)
    # 存下來，之後查東西不用重訓
    np.savez_compressed(f'lda_k{k}_sw.npz', phi=lda.phi, theta=lda.theta,
                        vocab=np.array(lda.vocabulary, dtype=object),
                        n_dt=lda.n_dt, doc_lengths=lda.doc_lengths,
                        alpha=lda.alpha, beta=np.array(lda.beta))
    token_share = lda.n_t / lda.n_t.sum()
    theta = lda.theta
    dominant = np.bincount(theta.argmax(axis=1), minlength=k) / len(theta)
    print(f'\n【 k={k} 】（依主題編號排列）')
    print(f'{"主題":>4} | token占比 | 最主要主題的詩篇占比')
    for t in range(k):
        print(f'{t:>4} | {token_share[t]:>8.1%} | {dominant[t]:>8.1%}')
    print(f'佔比範圍: token {token_share.min():.1%} ~ {token_share.max():.1%}')
