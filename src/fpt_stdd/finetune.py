from __future__ import annotations
import argparse, json
from pathlib import Path
from .data import load_finetune_pairs,select_ablation_training_frame

def main():
    p=argparse.ArgumentParser(description="Optional augmented SBERT fine-tuning experiment")
    p.add_argument("--strict-split",default="data/v1_1/splits/strict_group_split_v1_1.csv")
    p.add_argument("--synthetic",default="data/v1_1/augmentation/synthetic_duplicate_train_v1_1.csv")
    p.add_argument("--hard-negative",default="data/v1_1/augmentation/hard_negative_train_v1_1.csv")
    p.add_argument("--model",default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    p.add_argument("--output-dir",default="models/sbert_augmented"); p.add_argument("--epochs",type=int,default=3)
    p.add_argument("--batch-size",type=int,default=8); p.add_argument("--lr",type=float,default=2e-5); p.add_argument("--seed",type=int,default=2026)
    p.add_argument("--mode",choices=["e0_real_only","e1_positive_only","e2_hard_only","e3_combined"],default="e3_combined")
    args=p.parse_args()
    from datasets import Dataset
    from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses
    use_syn=args.mode in ("e1_positive_only","e3_combined")
    use_hard=args.mode in ("e2_hard_only","e3_combined")
    df=load_finetune_pairs(args.strict_split,args.synthetic,args.hard_negative,
        include_synthetic=use_syn,include_hard_negative=use_hard)
    syn=df[df.source=="synthetic_positive"].copy()
    train=select_ablation_training_frame(df,args.mode,args.seed)
    syn_selected=train[train.source=="synthetic_positive"]
    train["label"]=train.y.astype(float)
    ds=Dataset.from_pandas(train[["text_a","text_b","label"]],preserve_index=False)
    model=SentenceTransformer(args.model); loss=losses.ContrastiveLoss(model=model)
    ta=SentenceTransformerTrainingArguments(output_dir=args.output_dir,num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,learning_rate=args.lr,warmup_ratio=0.1,
        fp16=False,bf16=False,seed=args.seed,save_strategy="epoch",logging_steps=5)
    trainer=SentenceTransformerTrainer(model=model,args=ta,train_dataset=ds,loss=loss); trainer.train(); model.save(args.output_dir)
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    (Path(args.output_dir)/"training_manifest.json").write_text(json.dumps({"mode":args.mode,"seed":args.seed,"rows":len(train),
      "source_counts":train.source.value_counts().to_dict(),"selected_synthetic_id":syn_selected.annotation_id.tolist(),
      "effective_independent_positive_pairs":int(syn.group_id.nunique()) if len(syn) else 0,
      "hard_negative_share_of_all_negatives":float((train.source=="hard_negative").sum()/max(1,(train.y==0).sum())),
      "warning":"Synthetic rows are correlated views of verified train positives. Report mean±SD across seeds and only as an ablation."},indent=2),encoding="utf-8")

if __name__=="__main__": main()
