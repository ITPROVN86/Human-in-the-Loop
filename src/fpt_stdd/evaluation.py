from __future__ import annotations
import json, random
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score,balanced_accuracy_score,confusion_matrix,
    f1_score,fbeta_score,matthews_corrcoef,precision_score,recall_score,roc_auc_score)

def set_seed(seed): random.seed(seed); np.random.seed(seed)

def choose_threshold(y,scores,objective="f1"):
    y=np.asarray(y); scores=np.asarray(scores)
    candidates=np.unique(np.r_[scores,0.0,1.0])
    candidates=np.unique(np.r_[candidates,(candidates[:-1]+candidates[1:])/2])
    best=None
    for t in candidates:
        pred=(scores>=t).astype(int)
        primary=f1_score(y,pred,zero_division=0) if objective=="f1" else fbeta_score(y,pred,beta=2,zero_division=0)
        key=(primary,precision_score(y,pred,zero_division=0),recall_score(y,pred,zero_division=0),t)
        if best is None or key>best[0]: best=(key,float(t))
    return best[1]

def metrics(y,scores,threshold):
    y=np.asarray(y,dtype=int); scores=np.asarray(scores,dtype=float); pred=(scores>=threshold).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    out={"n":len(y),"positives":int(y.sum()),"threshold":float(threshold),"tp":int(tp),"fp":int(fp),"tn":int(tn),"fn":int(fn),
         "precision":precision_score(y,pred,zero_division=0),"recall":recall_score(y,pred,zero_division=0),
         "f1":f1_score(y,pred,zero_division=0),"f2":fbeta_score(y,pred,beta=2,zero_division=0),
         "balanced_accuracy":balanced_accuracy_score(y,pred),"mcc":matthews_corrcoef(y,pred)}
    out["average_precision"]=average_precision_score(y,scores) if y.sum()>0 else None
    out["roc_auc"]=roc_auc_score(y,scores) if len(np.unique(y))==2 else None
    return {k:(float(v) if isinstance(v,(np.floating,float)) else v) for k,v in out.items()}

def save_run(out_dir,protocol,baseline,split,df,scores,threshold,threshold_source):
    path=Path(out_dir)/protocol/baseline; path.mkdir(parents=True,exist_ok=True)
    pred=df[["annotation_id","topic_a_id","topic_b_id","semester_pair","label","split"]].copy()
    pred["score"]=scores; pred["threshold"]=threshold; pred["prediction"]=np.where(scores>=threshold,"Duplicate","Non-duplicate")
    pred.to_csv(path/f"predictions_{split}.csv",index=False)
    result=metrics(df.y,scores,threshold); result.update({"protocol":protocol,"baseline":baseline,"split":split,"threshold_source":threshold_source})
    (path/f"metrics_{split}.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    return result

