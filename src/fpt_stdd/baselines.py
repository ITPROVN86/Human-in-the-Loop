from __future__ import annotations
import math, re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

class TfidfPairScorer:
    name="tfidf"
    def __init__(self):
        self.word=TfidfVectorizer(lowercase=True,ngram_range=(1,2),sublinear_tf=True,min_df=1,max_features=30000)
        self.char=TfidfVectorizer(lowercase=True,analyzer="char_wb",ngram_range=(3,5),sublinear_tf=True,min_df=1,max_features=40000)
    def fit(self,texts):
        texts=list(texts); self.word.fit(texts); self.char.fit(texts); return self
    def score(self,a,b):
        aw,bw=self.word.transform(a),self.word.transform(b)
        ac,bc=self.char.transform(a),self.char.transform(b)
        sw=np.asarray(aw.multiply(bw).sum(axis=1)).ravel()
        sc=np.asarray(ac.multiply(bc).sum(axis=1)).ravel()
        return 0.65*sw+0.35*sc

def _tokens(text):
    return re.findall(r"[\w]+",str(text).lower(),flags=re.UNICODE)

class BM25PairScorer:
    name="bm25"
    def __init__(self,k1=1.5,b=0.75): self.k1=k1; self.b=b
    def fit(self,texts):
        docs=[_tokens(x) for x in texts]; self.n=len(docs); self.avgdl=sum(map(len,docs))/max(1,self.n)
        df=Counter()
        for d in docs: df.update(set(d))
        self.idf={t:math.log(1+(self.n-n+0.5)/(n+0.5)) for t,n in df.items()}; return self
    def _raw(self,query,doc):
        q=set(_tokens(query)); d=_tokens(doc); tf=Counter(d); dl=max(1,len(d)); total=0.0
        for term in q:
            f=tf.get(term,0); denom=f+self.k1*(1-self.b+self.b*dl/max(1,self.avgdl))
            if f: total+=self.idf.get(term,0.0)*f*(self.k1+1)/denom
        return total
    def score(self,a,b):
        out=[]
        for x,y in zip(a,b):
            xy=self._raw(x,y); yx=self._raw(y,x)
            xx=max(self._raw(x,x),1e-12); yy=max(self._raw(y,y),1e-12)
            out.append(0.5*(xy/math.sqrt(xx*yy)+yx/math.sqrt(xx*yy)))
        return np.clip(np.asarray(out,dtype=float),0,1)

class SBERTPairScorer:
    name="sbert"
    def __init__(self,model_name,device=None,batch_size=32):
        self.model_name=model_name; self.device=device; self.batch_size=batch_size
    def fit(self,texts):
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer(self.model_name,device=self.device); return self
    def score(self,a,b):
        all_text=list(dict.fromkeys(list(a)+list(b)))
        emb=self.model.encode(all_text,batch_size=self.batch_size,normalize_embeddings=True,show_progress_bar=True)
        idx={t:i for i,t in enumerate(all_text)}
        return np.asarray([float(np.dot(emb[idx[x]],emb[idx[y]])) for x,y in zip(a,b)])

def make_scorer(name,model_name=None,device=None,batch_size=32):
    if name=="tfidf": return TfidfPairScorer()
    if name=="bm25": return BM25PairScorer()
    if name=="sbert": return SBERTPairScorer(model_name,device,batch_size)
    raise ValueError(f"Unknown baseline {name!r}")

