# FPT-STDD-2026 Baselines

Mã nguồn tái lập cho bài toán phát hiện hai đề tài khóa luận có trùng lặp cấu trúc/ngữ nghĩa hay không.

## Baseline nên chạy

| ID | Mô hình | Vai trò trong paper | Fine-tune? |
|---|---|---|---|
| B0 | TF-IDF word + character n-gram cosine | Baseline lexical bắt buộc, nhanh và dễ tái lập | Không |
| B1 | BM25 đối xứng | Baseline lexical retrieval | Không |
| B2 | `paraphrase-multilingual-MiniLM-L12-v2` cosine | Baseline chính, hỗ trợ văn bản đa ngôn ngữ | Không |
| E1 | B2 fine-tune với positive augmentation + hard negatives | Thí nghiệm mở rộng/ablation, không gọi là baseline thuần | Có |

**Khuyến nghị chính:** dùng B2 làm baseline neural chính; B0 và B1 là các mốc lexical. Không dùng Logistic Regression/Random Forest làm baseline chính vì strict-train chỉ có **1 cặp dương độc lập**; mô hình supervised sẽ chủ yếu học từ các biến thể tổng hợp của cùng một seed.

Tài liệu nền:

- https://sbert.net/docs/sentence_transformer/usage/semantic_textual_similarity.html
- https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
- https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html

## Giao thức chọn lọc

1. `Strict_Group_Split` là kết quả chính: train/validation/test không trùng topic.
2. `Temporal_Split` kiểm tra tổng quát hóa sang FA26.
3. Không chọn threshold trên test. Threshold của mọi baseline được chọn trên strict validation, rồi khóa lại cho strict test và temporal test.
4. Báo cáo Precision, Recall, F1, F2, MCC, Balanced Accuracy, Average Precision và TP/FP/TN/FN.
5. Không gộp 24 positive augmentation thành 24 cặp dương độc lập.
6. Hard negatives chỉ xuất hiện trong train; validation/test không thay đổi.

Validation chỉ có 1 positive và strict test có 5 positives. Vì vậy phải công bố TP/FP/TN/FN bên cạnh F1; không báo cáo Accuracy đơn độc.

## Cấu trúc

```text
fpt_stdd_baselines/
├── configs/default.yaml
├── data/FPT_STDD_2026_...xlsx
├── scripts/
├── src/fpt_stdd/
│   ├── data.py          # đọc workbook, ghép topic text, dựng split
│   ├── baselines.py     # TF-IDF, BM25, Sentence-BERT
│   ├── evaluation.py    # threshold và metrics
│   ├── runner.py        # strict + temporal
│   ├── cli.py           # CLI baseline
│   └── finetune.py      # thí nghiệm E1 tùy chọn
├── tests/test_smoke.py
├── pyproject.toml
└── requirements.txt
```

## Cài đặt

### Windows PowerShell

```powershell
cd fpt_stdd_baselines
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

### Linux/WSL

```bash
cd fpt_stdd_baselines
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Lần đầu chạy B2, chương trình tải model từ Hugging Face. B0/B1 chạy CPU và không cần tải model.

## Cài môi trường GPU NVIDIA — Windows 10/11

### 1. Kiểm tra driver và GPU

Mở PowerShell hoặc Command Prompt:

```powershell
nvidia-smi
```

Nếu lệnh không tồn tại hoặc không hiển thị GPU, cần cài/cập nhật NVIDIA Driver trước. Không cần tự cài CUDA Toolkit nếu dùng PyTorch wheel vì wheel đã chứa CUDA runtime cần thiết.

### 2. Cài tự động

Mở PowerShell tại thư mục project:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_gpu_windows.ps1
```

Script tạo Python 3.11 virtual environment, cài PyTorch CUDA 12.8, cài project và chạy kiểm tra GPU.

## Activate môi trường mỗi lần mở terminal

Sau khi cài xong, **mỗi lần đóng rồi mở lại terminal đều phải vào đúng thư mục project và activate `.venv`**.

### PowerShell

```powershell
cd DUONG_DAN\fpt_stdd_baselines
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

### Command Prompt — CMD

```bat
cd /d DUONG_DAN\fpt_stdd_baselines
.venv\Scripts\activate.bat
```

### Git Bash trên Windows

```bash
cd /duong-dan/fpt_stdd_baselines
source .venv/Scripts/activate
```

### Ubuntu/WSL2

```bash
cd /duong-dan/fpt_stdd_baselines
source .venv/bin/activate
```

Khi activate thành công, đầu dòng lệnh xuất hiện `(.venv)`:

```text
(.venv) PS D:\Research\fpt_stdd_baselines>
```

Kiểm tra đang dùng đúng Python của môi trường:

```powershell
python -c "import sys; print(sys.executable)"
python scripts/check_gpu.py
```

Đường dẫn Python phải chứa `fpt_stdd_baselines\.venv`. Để thoát môi trường:

```powershell
deactivate
```

Nếu PowerShell báo `running scripts is disabled`, chạy lệnh sau trong đúng cửa sổ PowerShell hiện tại rồi activate lại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Chạy một lệnh tự động activate + kiểm tra GPU + chạy baseline

PowerShell:

```powershell
.\scripts\run_gpu_windows.ps1
```

CMD:

```bat
scripts\run_gpu_windows.bat
```

Hai script trên tự activate trong phiên chạy script; khi quay lại terminal và muốn gõ lệnh thủ công, vẫn activate bằng một trong các lệnh phía trên.

### 3. Cài thủ công

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# PyTorch GPU CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Các thư viện còn lại và source code
pip install -e .

# Kiểm tra PyTorch có nhận GPU
python scripts/check_gpu.py
```

Kết quả đúng phải có:

```text
CUDA available: True
GPU count: 1
GPU 0: NVIDIA ...
CUDA matrix test: PASS
```

PyTorch hiện cung cấp nhiều CUDA wheels. Nếu `cu128` không phù hợp với driver/GPU, chọn đúng lệnh tại trang chính thức: https://pytorch.org/get-started/locally/

## Cài môi trường GPU — Ubuntu/WSL2

Trước tiên, chạy `nvidia-smi` bên trong Ubuntu/WSL. Sau đó:

```bash
chmod +x scripts/setup_gpu_linux.sh
./scripts/setup_gpu_linux.sh
```

Hoặc cài thủ công:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -e .
python scripts/check_gpu.py
```

Với WSL2, chỉ cài NVIDIA Windows driver hỗ trợ WSL; không cài Linux NVIDIA driver đè lên driver do WSL cung cấp. Hướng dẫn chính thức: https://docs.nvidia.com/cuda/wsl-user-guide/index.html

## Chạy baseline bằng GPU

Chỉ Sentence-BERT sử dụng GPU; TF-IDF và BM25 vẫn chạy CPU.

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m fpt_stdd.cli --config configs/default.yaml --baselines sbert --device cuda --batch-size 32
```

```bash
# Linux/WSL
source .venv/bin/activate
python -m fpt_stdd.cli --config configs/default.yaml --baselines sbert --device cuda --batch-size 32
```

Chạy toàn bộ B0-B2:

```bash
python -m fpt_stdd.cli --config configs/default.yaml --device cuda --batch-size 32
```

Gợi ý batch size:

| VRAM | `--batch-size` khuyến nghị |
|---:|---:|
| 4 GB | 8 |
| 6-8 GB | 16-32 |
| 10-12 GB | 32-64 |
| 16 GB trở lên | 64-128 |

Nếu xuất hiện `CUDA out of memory`, giảm batch size xuống một nửa. Với model MiniLM này, batch size chỉ ảnh hưởng tốc độ và bộ nhớ khi inference, không thay đổi threshold protocol.

### Fine-tune bằng GPU

`SentenceTransformerTrainer` tự dùng CUDA khi `torch.cuda.is_available()` là `True`:

```bash
python -m fpt_stdd.finetune --workbook data/FPT_STDD_2026_Positive_Augmentation_and_Hard_Negatives_Train_Only.xlsx --output-dir models/sbert_augmented --epochs 3 --batch-size 8
```

Theo dõi GPU ở terminal khác:

```bash
nvidia-smi -l 2
```

Không nên tăng epoch trước khi chạy ablation 1/3/5 epochs vì tập dương độc lập rất nhỏ và dễ overfit.

## Chạy baseline

Chạy B0 và B1 trước:

```bash
python -m fpt_stdd.cli --config configs/default.yaml --baselines tfidf bm25 --device cpu
```

Chạy đủ B0-B2:

```bash
python -m fpt_stdd.cli --config configs/default.yaml
```

GPU NVIDIA:

```bash
python -m fpt_stdd.cli --config configs/default.yaml --device cuda
```

Chỉ chạy strict topic-disjoint với B2:

```bash
python -m fpt_stdd.cli --config configs/default.yaml --protocols strict --baselines sbert --text-view title_objectives
```

## Ablation text view

```bash
# A0: title tiếng Anh
python -m fpt_stdd.cli --config configs/default.yaml --text-view title

# A1: title EN + VI
python -m fpt_stdd.cli --config configs/default.yaml --text-view title_bilingual

# A2: title EN + VI + objectives — cấu hình chính
python -m fpt_stdd.cli --config configs/default.yaml --text-view title_objectives

# A3: toàn bộ proposal
python -m fpt_stdd.cli --config configs/default.yaml --text-view full
```

Không mặc định xem `full` là tốt nhất: multilingual MiniLM giới hạn chiều dài đầu vào nên proposal dài có thể bị cắt. `title_objectives` cân bằng tín hiệu ngữ nghĩa và độ dài.

## Output

```text
results/baselines/
├── metrics_summary.csv
├── run_metadata.json
├── strict/<baseline>/
│   ├── metrics_calibration.json
│   ├── metrics_test.json
│   ├── predictions_calibration.csv
│   └── predictions_test.csv
└── temporal/<baseline>/...
```

`metrics_summary.csv` dùng cho bảng paper; `predictions_test.csv` dùng cho error analysis.

### Kết quả kiểm tra B0/B1 đã chạy kèm project

Với `text_view=title_objectives` và threshold lấy từ strict validation:

| Protocol | Baseline | Precision | Recall | F1 | MCC | AP | TP/FP/FN |
|---|---|---:|---:|---:|---:|---:|---|
| Strict test | TF-IDF | 0.667 | 0.400 | 0.500 | 0.503 | 0.620 | 2/1/3 |
| Strict test | BM25 | 0.214 | 0.600 | 0.316 | 0.321 | 0.227 | 3/11/2 |
| Temporal FA26 | TF-IDF | 0.160 | 0.667 | 0.258 | 0.312 | 0.294 | 4/21/2 |
| Temporal FA26 | BM25 | 0.068 | 0.667 | 0.123 | 0.189 | 0.181 | 4/55/2 |

Đây là kết quả sanity check, không phải kết luận cuối cùng. B2 cần được chạy trong đúng môi trường báo cáo để ghi lại model revision, GPU/CPU và package versions.

## Fine-tune tùy chọn: E1

```bash
python -m fpt_stdd.finetune --workbook data/FPT_STDD_2026_Positive_Augmentation_and_Hard_Negatives_Train_Only.xlsx --output-dir models/sbert_augmented --epochs 3 --batch-size 8
```

Sau đó đổi trong `configs/default.yaml`:

```yaml
sbert_model: models/sbert_augmented
```

E1 chỉ là ablation vì synthetic positives bắt nguồn từ một cặp dương thật. Cần so sánh B2 zero-shot, B2 + positive augmentation và B2 + positive augmentation + hard negatives. Nếu validation tăng mạnh nhưng temporal test giảm, xem đó là dấu hiệu overfit vào seed dương.

## Component ablation E0–E3 với 5 seed

Định nghĩa thống nhất:

| Mode | Dữ liệu train |
|---|---|
| `e0_real_only` | 56 cặp strict-train thật |
| `e1_positive_only` | E0 + một synthetic positive view được chọn theo seed |
| `e2_hard_only` | E0 + 20 hard/semi-hard negatives |
| `e3_combined` | E0 + positive view + 20 hard/semi-hard negatives |

E1 chỉ lấy một synthetic view trong mỗi lần chạy vì toàn bộ 24 views bắt nguồn từ một cặp dương độc lập. Việc chạy 5 seed làm thay đổi view được chọn và đo độ ổn định mà không giả định 24 views là 24 Ground Truth độc lập.

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
.\scripts\run_component_ablation_5seeds.ps1
```

Linux/WSL:

```bash
source .venv/bin/activate
chmod +x scripts/run_component_ablation_5seeds.sh
./scripts/run_component_ablation_5seeds.sh
```

Quy trình sẽ chạy 20 lần fine-tune: 4 modes × 5 seeds. Kết quả tổng hợp:

```text
results/component_ablation_summary.csv
results/component_ablation_summary_all_runs.csv
```

Báo cáo mean ± standard deviation cho F1, MCC, Average Precision và ROC-AUC trên strict và temporal test. Không chọn seed tốt nhất để đưa vào paper.

## Bảng nên đưa vào paper

| Protocol | Method | Text view | Precision | Recall | F1 | F2 | MCC | AP | TP/FP/FN |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Strict topic-disjoint | TF-IDF | title+objectives | | | | | | | |
| Strict topic-disjoint | BM25 | title+objectives | | | | | | | |
| Strict topic-disjoint | mMiniLM zero-shot | title+objectives | | | | | | | |
| Temporal FA26 | TF-IDF | title+objectives | | | | | | | |
| Temporal FA26 | BM25 | title+objectives | | | | | | | |
| Temporal FA26 | mMiniLM zero-shot | title+objectives | | | | | | | |

## Tái lập và kiểm thử

- Seed: `2026`.
- Threshold luôn lấy từ strict validation.
- Vectorizer chỉ fit trên train text.
- Test không tham gia fit vocabulary, threshold hoặc fine-tune.
- Workbook đầu vào đã kèm trong `data/`.

```bash
pip install pytest
pytest -q
```
