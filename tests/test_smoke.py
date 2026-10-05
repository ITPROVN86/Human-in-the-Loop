import numpy as np
from fpt_stdd.baselines import TfidfPairScorer,BM25PairScorer
from fpt_stdd.evaluation import choose_threshold,metrics

def test_scorers_and_metrics():
    corpus=["restaurant management system","restaurant operations platform","public bus tracking"]
    a=[corpus[0],corpus[0]]; b=[corpus[1],corpus[2]]
    for scorer in (TfidfPairScorer(),BM25PairScorer()):
        scores=scorer.fit(corpus).score(a,b)
        assert len(scores)==2 and np.isfinite(scores).all()
    y=np.array([1,0]); t=choose_threshold(y,scores); out=metrics(y,scores,t)
    assert out["n"]==2 and 0<=out["f1"]<=1

