import pandas as pd, json, re, itertools, networkx as nx, nltk
from nltk.corpus import stopwords
from tqdm import tqdm

STOP_EN = set(stopwords.words('english'))
STOP_CN = set('的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行现所日力经里水化高自二理起小物实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月军公更拉东者集管处己将无战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严首底液官德随病苏失尔死讲配女黄推显谈罪神艺呢席含企望密批营项防举球英氧势告李台落木帮轮"

def clean(text):
    text = re.sub(r'[^ \u4e00-\u9fa5a-zA-Z]+',' ', text)
    tokens = nltk.word_tokenize(text.lower())
    return [w for w in tokens if w not in STOP_EN and w not in STOP_CN and len(w)>1]

# 假设评论已爬成 jsonl: {'movie':'xxx','lang':'cn','text':'...'}
df = pd.read_json('comments.jsonl', lines=True)
records = []
for _, row in tqdm(df.iterrows(), total=len(df)):
    tokens = clean(row['text'])
    emo = nrc_score(' '.join(tokens))   # 复用上一脚本函数
    records.append({'movie':row['movie'],
                    'lang':row['lang'],
                    'tokens':tokens,
                    'nrc':emo})

# 共现网络：全语料按窗口=1
G = nx.Graph()
for r in records:
    for w1,w2 in itertools.combinations(r['tokens'],2):
        if G.has_edge(w1,w2):
            G[w1][w2]['weight']+=1
        else:
            G.add_edge(w1,w2,weight=1)
# 过滤低频边
G.remove_edges_from([(u,v,d) for u,v,d in G.edges(data=True) if d['weight']<3])
G.remove_nodes_from(list(nx.isolates(G)))
nx.write_gexf(G,'comment_cooc.gexf')
json.dump(records, open('comment_clean.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
print('✅ 评论清洗+共现网络完成')