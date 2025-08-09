import re, json, pickle, torch, nltk
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from pathlib import Path

SRC_DIR  = 'subtitles'   # 原始 .srt/.ass
OUT_JSON = 'subtitle_emotion.jsonl'

# 加载情感词典
nrc = pickle.load(open('nrc_emo_dict.pkl','rb'))
emo_list = ['anger','anticipation','disgust','fear','joy','sadness','surprise','trust']

# 预训练模型
tok  = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert = BertModel.from_pretrained('bert-base-multilingual-cased').eval()

def srt2lines(path):
    with open(path, encoding='utf-8', errors='ignore') as f:
        raw = f.read()
    # 简单正则提取文本行
    lines = re.findall(r'\d+\n.*?\n(.*)', raw, flags=re.S)
    return [l.replace('\n',' ').strip() for l in lines if l.strip()]

def nrc_score(text):
    tokens = nltk.word_tokenize(text.lower())
    vec = {e:0 for e in emo_list}
    for w in tokens:
        if w in nrc:
            for e in emo_list:
                vec[e] += nrc[w].get(e,0)
    return vec

def bert_vec(text):
    ids = tok(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        vec = bert(**ids).last_hidden_state.mean(dim=1).squeeze().numpy()
    return vec.tolist()

results = []
for f in Path(SRC_DIR).glob('*'):
    lines = srt2lines(f)
    for t in lines:
        res = {'movie':f.stem,'text':t,
               'nrc':nrc_score(t),
               'bert_vec':bert_vec(t)}
        results.append(res)
json.dump(results, open(OUT_JSON,'w',encoding='utf-8'), ensure_ascii=False, indent=1)
print('✅ 字幕情感+向量完成')