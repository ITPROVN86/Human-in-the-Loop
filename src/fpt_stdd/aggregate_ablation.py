from __future__ import annotations
import argparse,glob,re
from pathlib import Path
import pandas as pd

def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",default="results/component_ablation"); p.add_argument("--output",default="results/component_ablation_summary.csv"); args=p.parse_args()
    rows=[]
    for f in glob.glob(str(Path(args.root)/"*"/"seed_*"/"metrics_summary.csv")):
        path=Path(f); mode=path.parent.parent.name; seed=int(path.parent.name.split('_')[-1]); d=pd.read_csv(path); d=d[d.split=='test'].copy(); d['mode']=mode; d['seed']=seed; rows.append(d)
    if not rows: raise SystemExit(f"No metrics_summary.csv found under {args.root}")
    all_runs=pd.concat(rows,ignore_index=True); metrics=['precision','recall','f1','f2','mcc','average_precision','roc_auc','tp','fp','fn']
    summary=all_runs.groupby(['mode','protocol','baseline'])[metrics].agg(['mean','std','min','max']).reset_index()
    summary.columns=['_'.join([str(x) for x in c if x]).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns]
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); summary.to_csv(out,index=False); all_runs.to_csv(out.with_name(out.stem+'_all_runs.csv'),index=False)
    print(summary.to_string(index=False))

if __name__=="__main__": main()

