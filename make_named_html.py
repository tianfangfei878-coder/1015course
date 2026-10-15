"""為 LDA 主題命名（含信心標記；無把握者不命名並註明原因），輸出 HTML + 命名對照 CSV
- 資料來源：lda_k10_sw.npz / lda_k20_sw.npz（先前訓練結果，不重訓）
- 命名依據：主題前 15 字 + 最典型詩的內容
"""
import json, csv, html
import numpy as np

with open('全唐诗.json') as f:
    poems = json.load(f)
meta = []
for p in poems:
    text = ''.join(p['paragraphs'])
    if text:
        meta.append((p['title'], p['author'] or '（佚名）', text))

# ---------------- 命名（name=None 表示無法命名） ----------------
NAMES = {
 10: {
  0: ('江湖行旅', '高', ''),
  1: ('贈答遣懷', '中', '贈答與閒居詩酒交疊，難截然分開'),
  2: ('鞍馬行役', '中', '邊塞（涼州詞）、輓歌、送別皆入此主題，共通點主要在鞍馬道路意象'),
  3: ('郊廟頌聖', '高', ''),
  4: ('詠物諧趣', '中', '用字俚俗（一、頭、兒、白），典型詩多為詠物打油小詩，涵蓋面不易確定'),
  5: ('擬古玄言', '中', '典型詩高度集中於元結擬古騷體（兮字句），可能偏狹'),
  6: ('幽居清賞', '高', ''),
  7: ('遊仙慕道', '高', ''),
  8: ('山寺僧禪', '高', ''),
  9: ('花艷閨情', '中', '詠花與閨豔交疊'),
 },
 20: {
  0: ('登望山川', '高', ''),
  1: ('山寺禪隱', '高', ''),
  2: ('詩酒閒適', '高', ''),
  3: ('秋夜寒寂', '高', ''),
  4: ('春柳離情', '中', '春景與離情交疊，與主題 12、14 邊界模糊'),
  5: ('邊塞征戰', '高', ''),
  6: ('郊廟雅樂', '高', ''),
  7: ('江湖客旅', '高', ''),
  8: ('說理詠懷', '中', '議論說理之作，作者風格差異大（元結、孟郊、白居易）'),
  9: (None, '低', '混雜贈答、留別、城中官署往還；字面（城、門、書、馬、朝、路）無一致意象，不予命名'),
  10: ('龍雷險怪', '中', '龍雷風雨奇象，多韓愈一派險怪之作，樣本似偏少'),
  11: ('賦得詠物', '中', '「賦得」試帖與清景詠物混合'),
  12: ('惜春傷別', '中', '惜春、傷逝、折柳送別交疊'),
  13: ('楚辭騷體', '高', ''),
  14: ('離別相思', '高', ''),
  15: (None, '低', '多為口語疊句（「一來一回」）與登臨古蹟混雜，意象分歧，不予命名'),
  16: ('宮廷應制', '高', ''),
  17: ('田家村舍', '高', ''),
  18: ('花間豔曲', '高', ''),
  19: ('禪道玄思', '中', '禪僧、丹道、病中參悟混雜'),
 },
}

# ---------------- 組 HTML ----------------
css = """body{font-family:"PingFang TC","Microsoft JhengHei",serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#222}
h1{font-size:26px}h2{border-bottom:2px solid #8b0000;padding-bottom:4px;margin-top:40px}
.meta{color:#666;font-size:13px}
table.sum{border-collapse:collapse;font-size:13px;margin:10px 0 26px}
table.sum th,table.sum td{border:1px solid #ddd;padding:4px 10px;text-align:left}
table.sum th{background:#f3ede2}
.conf{display:inline-block;font-size:11px;padding:1px 7px;border-radius:8px;margin-top:4px}
.conf-高{background:#e3f2e3;color:#1b5e20}.conf-中{background:#fff3d6;color:#8a6d00}.conf-低{background:#fde3e3;color:#a11}
.unote{font-size:12px;color:#a11;margin-top:6px;max-width:250px}
.topic{display:flex;gap:14px;margin:14px 0;padding:12px;background:#faf7f2;border:1px solid #e5ddcf;border-radius:8px;align-items:flex-start}
.topic.unk{background:#fdf3f3;border-color:#ecc8c8}
.tid{min-width:100px;font-weight:bold;color:#8b0000;padding-top:6px}
.tname{display:block;font-size:17px;color:#222;margin-top:2px}
.tname-unk{color:#a11}
.words{min-width:280px;line-height:2}
.w{display:inline-block;margin-right:10px;color:#333}
.poem{flex:1;background:#fff;border-left:3px solid #c9a86a;padding:6px 10px;margin:4px 0;font-size:14px}
.ptitle{font-weight:600;font-size:13px}.theta{color:#888;font-weight:normal;font-size:12px;margin-left:8px}
.ptext{margin-top:4px;line-height:1.7}
legend{font-size:12px;color:#666}"""

html_blocks, csv_rows = [], []
for k in (10, 20):
    d = np.load(f'lda_k{k}_sw.npz', allow_pickle=True)
    phi, theta, vocab = d['phi'], d['theta'], d['vocab']
    n_dt, doc_lengths = d['n_dt'], d['doc_lengths']
    token_share = n_dt.sum(axis=0) / n_dt.sum()
    dominant = np.bincount(theta.argmax(axis=1), minlength=k) / len(theta)

    block = [f'<h2>{k} 個主題</h2>',
             '<table class="sum"><tr><th>主題</th><th>命名</th><th>信心</th><th>字數占比</th><th>詩篇占比*</th></tr>']
    for t in range(k):
        name, conf, note = NAMES[k][t]
        block.append(f'<tr><td>{t}</td><td>{name or "？（未命名）"}</td>'
                     f'<td><span class="conf conf-{conf}">{conf}</span></td>'
                     f'<td>{token_share[t]:.1%}</td><td>{dominant[t]:.1%}</td></tr>')
    block.append('</table><legend>*「詩篇占比」= 以該主題為最主要主題的詩篇數比例；字數占比 = 全語料字數中歸屬該主題的比例。'
                 '信心「低」者依指示不予命名。</legend>')

    for t in range(k):
        name, conf, note = NAMES[k][t]
        top_idx = np.argsort(-phi[t])[:10]
        tw = [(vocab[i], phi[t, i]) for i in top_idx]
        wmax = tw[0][1]
        words_html = ''.join(
            f'<span class="w" style="font-size:{12 + 22 * p / wmax:.0f}px">{html.escape(w)}</span>'
            for w, p in tw)
        name_html = (f'<span class="tname">{html.escape(name)}</span>' if name
                     else '<span class="tname tname-unk">？（未命名）</span>')
        note_html = f'<div class="unote">{html.escape(note)}</div>' if note else ''
        poem_cells = []
        for j, doc_id in enumerate(np.argsort(-theta[:, t])[:2], 1):
            title, author, text = meta[doc_id]
            prob = theta[doc_id, t]
            poem_cells.append(
                f'<div class="poem"><div class="ptitle">{j}. {html.escape(title)} ／ {html.escape(author)}'
                f'<span class="theta">θ={prob:.3f}</span></div>'
                f'<div class="ptext">{html.escape(text)}</div></div>')
        cls = 'topic unk' if not name else 'topic'
        block.append(f'<div class="{cls}"><div class="tid">主題 {t}{name_html}'
                     f'<span class="conf conf-{conf}">信心：{conf}</span>{note_html}</div>'
                     f'<div class="words">{words_html}</div>{"".join(poem_cells)}</div>')

        csv_rows.append({'n_topics': k, 'topic_id': t, 'name': name or '',
                         'confidence': conf, 'note': note,
                         'top10_words': '、'.join(w for w, _ in tw),
                         'token_share': round(float(token_share[t]), 4),
                         'dominant_doc_share': round(float(dominant[t]), 4),
                         'poem1_title': meta[np.argsort(-theta[:, t])[0]][0],
                         'poem2_title': meta[np.argsort(-theta[:, t])[1]][0]})
    html_blocks.append('\n'.join(block))

html_doc = ('<!DOCTYPE html><html lang="zh-Hant"><head><meta charset="utf-8">'
            '<title>《全唐詩》LDA 主題命名</title><style>' + css + '</style></head><body>'
            '<h1>《全唐詩》主題模型：主題命名</h1>'
            '<p class="meta">每首詩為一篇文件，以字為單位（去除標點），並以 <code>zh_cl_sim</code> 文言停用詞過濾；'
            'LDA Gibbs sampling，100 iterations，seed 42。命名由模型產出後人工判讀（依前 15 個主題字與典型詩內容），'
            '<b>信心「低」的主題依指示不予命名</b>，原因顯示於紅字。</p>'
            + '\n'.join(html_blocks) + '</body></html>')
with open('lda_results_named.html', 'w', encoding='utf-8') as f:
    f.write(html_doc)

with open('topic_names.csv', 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    w.writeheader()
    w.writerows(csv_rows)

print('輸出：lda_results_named.html、topic_names.csv')
for k in (10, 20):
    named = sum(1 for t in range(k) if NAMES[k][t][0])
    print(f'k={k}: 命名 {named}/{k}，未命名 {[t for t in range(k) if not NAMES[k][t][0]]}')
