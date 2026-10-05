from __future__ import annotations
from pathlib import Path
import pandas as pd

SHEET_BY_PROTOCOL = {"strict": "Strict_Group_Split", "temporal": "Temporal_Split"}

FINETUNE_COLUMNS = [
    "annotation_id", "topic_a_id", "topic_b_id", "text_a", "text_b",
    "y", "source", "sample_weight", "group_id",
]

def _read_excel(path: str | Path, sheet: str) -> pd.DataFrame:
    # read_only=False is intentional: artifact-generated XLSX files may omit sheet dimensions.
    return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")

def load_topics(path: str | Path) -> pd.DataFrame:
    df = _read_excel(path, "Topics_Standard").fillna("")
    if df["topic_id"].duplicated().any():
        raise ValueError("Topics_Standard contains duplicate topic_id values")
    return df.set_index("topic_id", drop=False)

def build_topic_text(row: pd.Series, view: str) -> str:
    fields = {
        "title": ["title_en_clean"],
        "title_bilingual": ["title_en_clean", "title_vi_clean"],
        "title_objectives": ["title_en_clean", "title_vi_clean", "objectives"],
        "full": ["title_en_clean", "title_vi_clean", "description_context", "objectives",
                 "scope_research_content", "functional_features", "expected_result"],
    }
    if view not in fields:
        raise ValueError(f"Unknown text_view={view!r}; choose one of {sorted(fields)}")
    values=[]
    for key in fields[view]:
        val=str(row.get(key, "") or "").strip()
        if val and val not in values:
            values.append(val)
    return "\n".join(values)

def load_protocol(path: str | Path, protocol: str, text_view: str) -> pd.DataFrame:
    if protocol not in SHEET_BY_PROTOCOL:
        raise ValueError(f"Unknown protocol {protocol!r}")
    pairs=_read_excel(path,SHEET_BY_PROTOCOL[protocol]).fillna("")
    topics=load_topics(path)
    missing=(set(pairs.topic_a_id)|set(pairs.topic_b_id))-set(topics.index)
    if missing:
        raise ValueError(f"Missing topic records: {sorted(missing)[:5]}")
    pairs=pairs.copy()
    pairs["text_a"]=[build_topic_text(topics.loc[t],text_view) for t in pairs.topic_a_id]
    pairs["text_b"]=[build_topic_text(topics.loc[t],text_view) for t in pairs.topic_b_id]
    pairs["y"]=(pairs["label"].str.strip().str.lower()=="duplicate").astype(int)
    pairs["protocol_name"]=protocol
    return pairs

def load_strict_csv(path: str | Path) -> pd.DataFrame:
    """Load the topic-disjoint v1.1 CSV without requiring the legacy workbook."""
    df = pd.read_csv(path).fillna("")
    required = {"annotation_id", "topic_a_id", "topic_b_id", "title_a_en",
                "title_b_en", "label", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Strict split CSV is missing columns: {sorted(missing)}")
    df = df.copy()
    df["text_a"] = df["title_a_en"].astype(str).str.strip()
    df["text_b"] = df["title_b_en"].astype(str).str.strip()
    numeric = pd.to_numeric(df["label"], errors="coerce")
    if numeric.isna().any() or not set(numeric.astype(int).unique()).issubset({0, 1}):
        raise ValueError("Strict split label must contain only numeric 0/1 values")
    df["y"] = numeric.astype(int)
    if "semester_pair" not in df.columns:
        df["semester_pair"] = [f"{str(a).split('_',1)[0]}-{str(b).split('_',1)[0]}"
                               for a,b in zip(df.topic_a_id,df.topic_b_id)]
    df["protocol_name"] = "strict"
    return df

def split_frames(df: pd.DataFrame, protocol: str) -> dict[str,pd.DataFrame]:
    if protocol=="strict":
        return {k:df[df.split==k].reset_index(drop=True) for k in ("train","validation","test")}
    if protocol=="temporal":
        return {"train_dev":df[df.split=="train_dev"].reset_index(drop=True),
                "test":df[df.split=="test_temporal"].reset_index(drop=True)}
    raise ValueError(protocol)

def load_finetune_pairs(
    strict_split: str | Path,
    synthetic_path: str | Path | None = None,
    hard_negative_path: str | Path | None = None,
    *,
    include_synthetic: bool = True,
    include_hard_negative: bool = True,
) -> pd.DataFrame:
    """Build the fine-tuning frame for E0-E3 from v1.1 CSV artifacts.

    Synthetic and hard-negative files are read only when their component is
    enabled, preventing E1/E2 from accidentally consuming the other component.
    """
    frames=[]
    strict=load_strict_csv(strict_split)
    train=strict[strict.split=="train"][["annotation_id","topic_a_id","topic_b_id","text_a","text_b","y"]].copy()
    train["source"]="real_strict_train"; train["sample_weight"]=1.0; train["group_id"]=train.annotation_id
    frames.append(train)
    train_topics=set(train.topic_a_id)|set(train.topic_b_id)

    if include_synthetic:
        if synthetic_path is None:
            raise ValueError("synthetic_path is required when include_synthetic=True")
        syn=pd.read_csv(synthetic_path).fillna("")
        syn=syn.rename(columns={"augmentation_id":"annotation_id","source_topic_a_id":"topic_a_id","source_topic_b_id":"topic_b_id"})
        required={"annotation_id","topic_a_id","topic_b_id","group_id","text_a","text_b","sample_weight"}
        missing=required-set(syn.columns)
        if missing: raise ValueError(f"Synthetic CSV is missing columns: {sorted(missing)}")
        syn_topics=set(syn.topic_a_id)|set(syn.topic_b_id)
        if syn_topics-train_topics:
            raise ValueError("Synthetic CSV contains a topic outside strict train")
        syn["y"]=1; syn["source"]="synthetic_positive"; syn["sample_weight"]=pd.to_numeric(syn.sample_weight)
        frames.append(syn[FINETUNE_COLUMNS])

    if include_hard_negative:
        if hard_negative_path is None:
            raise ValueError("hard_negative_path is required when include_hard_negative=True")
        hard=pd.read_csv(hard_negative_path).fillna("")
        hard=hard.rename(columns={"mining_id":"annotation_id","text_a_full":"text_a","text_b_full":"text_b","sampling_group":"group_id"})
        required={"annotation_id","topic_a_id","topic_b_id","group_id","text_a","text_b","sample_weight"}
        missing=required-set(hard.columns)
        if missing: raise ValueError(f"Hard-negative CSV is missing columns: {sorted(missing)}")
        hard_topics=set(hard.topic_a_id)|set(hard.topic_b_id)
        if hard_topics-train_topics:
            raise ValueError("Hard-negative CSV contains a topic outside strict train")
        hard["y"]=0; hard["source"]="hard_negative"; hard["sample_weight"]=pd.to_numeric(hard.sample_weight)
        frames.append(hard[FINETUNE_COLUMNS])
    return pd.concat(frames,ignore_index=True)

def select_ablation_training_frame(df: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    """Apply E0-E3 component selection, including one synthetic view per source pair."""
    valid={"e0_real_only","e1_positive_only","e2_hard_only","e3_combined"}
    if mode not in valid: raise ValueError(f"Unknown ablation mode {mode!r}; choose one of {sorted(valid)}")
    real=df[df.source=="real_strict_train"].copy()
    synthetic=df[df.source=="synthetic_positive"].copy()
    hard=df[df.source=="hard_negative"].copy()
    selected_syn=(synthetic.groupby("group_id",group_keys=False).sample(n=1,random_state=seed)
                  if len(synthetic) else synthetic)
    parts=[real]
    if mode in {"e1_positive_only","e3_combined"}: parts.append(selected_syn)
    if mode in {"e2_hard_only","e3_combined"}: parts.append(hard)
    return pd.concat(parts,ignore_index=True)
