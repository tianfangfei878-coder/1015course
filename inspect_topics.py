"""檢視每個主題：前 15 字 + 最典型 3 首（題名／作者／開頭），供命名判斷"""
import json
import numpy as np

with open('全唐诗.json') as f:
    poems = json.load(f)
meta = []
for p in poems:
    text = ''.join(p['paragraphs'])
    if text:
        meta.append((p['title'], p['author'] or '佚名', text))

for k in (10, 20):
    d = np.load(f'lda_k{k}_sw.npz', allow_pickle=True)
    phi, theta, vocab, n_dt, doc_lengths = d['phi'], d['theta'], d['vocab'], d['n_dt'], d['doc_lengths']
    print(f'\n===== k={k} =====')
    for t in range(k):
        top = np.argsort(-phi[t])[:15]
        words = '、'.join(vocab[i] for i in top)
        print(f'\n主題 {t}: {words}')
        for doc_id in np.argsort(-theta[:, t])[:3]:
            title, author, text = meta[doc_id]
            print(f'   θ={theta[doc_id, t]:.3f}  {title}／{author}  「{text[:60]}…」')
