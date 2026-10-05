from __future__ import annotations
import argparse, yaml
from pathlib import Path
from .runner import run

def main():
    p=argparse.ArgumentParser(description="Run FPT-STDD-2026 baselines")
    p.add_argument("--config",default="configs/default.yaml")
    p.add_argument("--workbook"); p.add_argument("--strict-split"); p.add_argument("--output-dir"); p.add_argument("--protocols",nargs="+")
    p.add_argument("--baselines",nargs="+"); p.add_argument("--text-view",choices=["title","title_bilingual","title_objectives","full"])
    p.add_argument("--device"); p.add_argument("--batch-size",type=int); p.add_argument("--seed",type=int); p.add_argument("--sbert-model"); p.add_argument("--threshold-objective",choices=["f1","f2"])
    args=p.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    for key in ("workbook","strict_split","output_dir","protocols","baselines","text_view","device","batch_size","seed","sbert_model","threshold_objective"):
        value=getattr(args,key,None)
        if value is not None: cfg[key]=value
    print(run(cfg).to_string(index=False))

if __name__=="__main__": main()
