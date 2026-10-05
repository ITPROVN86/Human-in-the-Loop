from __future__ import annotations
import json, platform, sys
from pathlib import Path
import pandas as pd
from .data import load_protocol,load_strict_csv,split_frames
from .baselines import make_scorer
from .evaluation import choose_threshold,save_run,set_seed

def _device(value):
    if value!="auto": return value
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception: return "cpu"

def run(config):
    set_seed(int(config.get("seed",2026))); out=Path(config["output_dir"]); out.mkdir(parents=True,exist_ok=True)
    workbook=config.get("workbook"); view=config.get("text_view","title_objectives")
    strict_csv=config.get("strict_split")
    strict=split_frames(load_strict_csv(strict_csv) if strict_csv else load_protocol(workbook,"strict",view),"strict")
    results=[]
    for protocol in config["protocols"]:
        if protocol=="strict" and strict_csv:
            frames=strict
        else:
            if not workbook:
                raise ValueError(f"Protocol {protocol!r} requires the legacy workbook")
            frames=split_frames(load_protocol(workbook,protocol,view),protocol)
        fit_frame=frames["train"] if protocol=="strict" else frames["train_dev"]
        calibration=strict["validation"]
        for name in config["baselines"]:
            scorer=make_scorer(name,config.get("sbert_model"),_device(config.get("device","auto")),int(config.get("batch_size",32)))
            fit_text=pd.concat([fit_frame.text_a,fit_frame.text_b]).drop_duplicates().tolist(); scorer.fit(fit_text)
            cal_scores=scorer.score(calibration.text_a.tolist(),calibration.text_b.tolist())
            threshold=choose_threshold(calibration.y,cal_scores,config.get("threshold_objective","f1"))
            results.append(save_run(out,protocol,name,"calibration",calibration,cal_scores,threshold,"strict_validation"))
            test=frames["test"]; scores=scorer.score(test.text_a.tolist(),test.text_b.tolist())
            results.append(save_run(out,protocol,name,"test",test,scores,threshold,"strict_validation"))
    summary=pd.DataFrame(results); summary.to_csv(out/"metrics_summary.csv",index=False)
    metadata={"python":sys.version,"platform":platform.platform(),"config":config}
    (out/"run_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")
    return summary
