# OBE-VLM-Grading

OBE-VLM-Grading is a reproducible research codebase for criterion-level grading of handwritten university exam answers with vision-language models. It implements four learning regimes, a cascaded ensemble, five evaluation studies, Study 5 explanation images, dataset generation, and the released numerical results.

Release: `2.0.1`

## Research scope

The repository evaluates these systems:

| Key | Model | Family | Approximate size |
|---|---|---|---:|
| `qwen25vl` | Qwen2.5-VL-7B-Instruct | Chat VLM | 7 B |
| `internvl3` | InternVL3-8B | Chat VLM | 8 B |
| `pixtral` | Pixtral-12B | Chat VLM | 12 B |
| `donut` | Donut base | OCR-free encoder-decoder | 0.2 B |
| `ensemble` | Cascaded Qwen, InternVL, and Pixtral system | Ensemble | n/a |

Each model is evaluated under four regimes:

| Regime | Description |
|---|---|
| `zero_shot` | Instruction and grading rubric only |
| `few_shot` | Instruction, rubric, and stratified graded examples |
| `lora` | LoRA fine-tuning with frozen base weights |
| `full` | Resource-aware partial fine-tuning for chat VLMs and full fine-tuning for Donut |

Donut is not instruction tuned and cannot consume in-context examples. Its zero-shot and few-shot results should be interpreted as a floor. For Donut, the few-shot prompt is the same as the zero-shot prompt.

The five studies cover:

1. Accuracy and criterion-level performance
2. AI-human and human-human agreement
3. Test-retest reliability
4. Subgroup fairness and error taxonomy
5. Attention-based explanations, deletion faithfulness, and teacher survey analysis

## Dataset

The default configuration loads `khalidsyfullah/obe-exam-grading` from the Hugging Face Hub. Each row represents one answer and one rubric criterion. Rows contain the answer image, question, reference answer, rubric, criterion maximum, and examiner score.

| Split | Criterion records | Distinct answers |
|---|---:|---:|
| Train | 1,483 | 363 |
| Validation | 208 | 50 |
| Test | 291 | 72 |
| Total | 1,982 | 485 |

The pipeline repairs unreliable `criterion_max` values using a lookup learned from the training split only. It also evaluates scores on a `0.05` mark grid so decimal scores such as `1.8` are preserved exactly. Validation and test labels are not used to construct prompts or score clamps.

## Repository layout

```text
obe-vlm-grading/
├── README.md
├── CHANGELOG.md
├── config.yaml
├── requirements.txt
├── data/
├── results/
├── notebooks/
│   └── OBE_Dataset_Generation.ipynb
├── src/
├── scripts/
├── tests/
└── Figure Generation Script/
```

The main pipeline writes only CSV and JSON result files. The released Study 5 PNG explanation images are retained. Files inside `data/` are retained in their original formats because they are study inputs and grading-pack assets.

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <repository-url>
cd obe-vlm-grading
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

On Windows Command Prompt:

```bat
py -m venv .venv
.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
```

Authenticate once if the dataset or model requires Hub access:

```bash
hf auth login
hf auth whoami
```

No Hugging Face token is stored in this repository. Hub operations use the credential cached by the Hugging Face CLI. Do not add credentials to `config.yaml`, notebooks, shell scripts, or source files.

## Quick validation

Validate the configuration and print the planned experiment sweep without loading a model:

```bash
python3 scripts/run_all.py --dry-run
```

Run the automated tests:

```bash
python3 -m unittest discover -s tests -v
```

The full test suite includes CPU pipeline tests and unit tests for training and XAI helpers. It does not download or run the large VLM checkpoints.

## Running experiments

Run one model and one regime:

```bash
python3 scripts/run_experiment.py --model qwen25vl --regime zero_shot
python3 scripts/run_experiment.py --model qwen25vl --regime few_shot
python3 scripts/run_experiment.py --model qwen25vl --regime lora
python3 scripts/run_experiment.py --model donut --regime full
```

Reuse an existing complete checkpoint:

```bash
python3 scripts/run_experiment.py --model qwen25vl --regime lora --skip-train
```

Run a short inference smoke test:

```bash
python3 scripts/run_experiment.py --model qwen25vl --regime zero_shot --limit 20
```

Build an ensemble after the three configured member models have completed the same regime:

```bash
python3 scripts/run_ensemble.py --regime lora
```

Run the configured sweep:

```bash
python3 scripts/run_all.py
python3 scripts/run_all.py --with-reliability --with-studies
python3 scripts/run_all.py --with-inspect
```

Useful sweep controls:

```bash
python3 scripts/run_all.py --models qwen25vl internvl3
python3 scripts/run_all.py --regimes zero_shot few_shot
python3 scripts/run_all.py --skip-train
python3 scripts/run_all.py --force
```

Completed outputs are detected automatically unless `--force` is used.

## Running the five studies

Study 1:

```bash
python3 scripts/study1_accuracy.py
```

Study 2:

```bash
python3 scripts/study2_human_agreement.py
```

`data/human_ratings.csv` contains the released ratings or serves as the second-rater template. Required columns are `example_id`, `criterion_index`, `rater_id`, and `score`.

Study 3 requires repeated stochastic predictions:

```bash
python3 scripts/run_experiment.py --model qwen25vl --regime lora --skip-train --reliability
python3 scripts/run_experiment.py --model internvl3 --regime lora --skip-train --reliability
python3 scripts/study3_reliability.py --regime lora
```

Study 4:

```bash
python3 scripts/study4_fairness_error.py
python3 scripts/study4_fairness_error.py --model qwen25vl --regime lora
```

Study 5 can be run in stages:

```bash
python3 scripts/study5_xai_eval.py --stage heatmaps
python3 scripts/study5_xai_eval.py --stage faithfulness
python3 scripts/study5_xai_eval.py --stage survey
```

The heatmap stage is the only analysis stage in the main pipeline that creates PNG assets. These PNG files are the Study 5 explanation images. Manuscript charts and tables are generated separately from the released result files.

## Dataset generation

The command-line dataset generator is the source of truth:

```bash
python3 scripts/generate_dataset.py \
  --data-dir /path/to/Processed_Dataset \
  --repo-id account/obe-exam-grading
```

The input folder must contain `train.json`, `val.json`, `test.json`, and `files/` with the referenced PDFs. The generator renders each PDF once, checks cross-split leakage, converts answer-level records to criterion-level rows, and uploads a private dataset by default.

To build locally instead of uploading:

```bash
python3 scripts/generate_dataset.py \
  --data-dir /path/to/Processed_Dataset \
  --output-dir /path/to/generated_dataset
```

`notebooks/OBE_Dataset_Generation.ipynb` provides the same workflow for Colab. Authentication uses the Hugging Face login widget only when uploading. The notebook contains no saved credential and no execution output.

## Dataset inspection and grading pack

Audit dataset structure, score grids, image statistics, and split leakage:

```bash
python3 scripts/inspect_dataset.py --sample-images 200
python3 scripts/inspect_dataset.py --sample-images 0
```

Create the optional human-grading pack under `data/grading_pack/`:

```bash
python3 scripts/export_grading_pack.py
```

The grading pack is part of `data/`, so its HTML, CSV, XLSX, and answer images are retained.

## Publication figures

`Figure Generation Script/` contains one self-contained Python file for each final manuscript figure. Each script reads its source files directly from `results/` and saves one PDF plus one PNG. Run these commands from the repository root:

```bash
python3 "Figure Generation Script/figure_4_accuracy.py"
python3 "Figure Generation Script/figure_5_human_agreement.py"
python3 "Figure Generation Script/figure_6_reliability.py"
python3 "Figure Generation Script/figure_7_fairness.py"
python3 "Figure Generation Script/figure_8_xai.py"
python3 "Figure Generation Script/figure_9_heatmaps.py"
```

Generated artifacts are written to `Figures/`, which is ignored by Git. Every script accepts `--results-dir` and `--output-dir`. No table-generation or diagram-generation scripts are included.

## Configuration and hyperparameters

`config.yaml` is the single source of truth. Pass another file with `--config path/to/config.yaml` or set `OBE_CONFIG`.

### Global and paths

| Key | Default | Meaning |
|---|---:|---|
| `seed` | `42` | Global random seed |
| `paths.output_root` | `results` | Results and checkpoint root |
| `paths.local_dataset_dir` | `dataset` | Local DatasetDict directory when `dataset.source` is `local` |
| `paths.human_ratings_csv` | `data/human_ratings.csv` | Study 2 human ratings |
| `paths.teacher_survey_csv` | `data/teacher_survey_responses.csv` | Study 5 teacher responses |

### Dataset

| Key | Default | Meaning |
|---|---|---|
| `dataset.source` | `hf_hub` | Dataset source, either `hf_hub` or `local` |
| `dataset.hf_repo` | `khalidsyfullah/obe-exam-grading` | Hub dataset ID |
| `dataset.splits` | train, validation, test | Required splits |
| `dataset.max_image_long_side` | `1280` | Default maximum image side |
| `dataset.cache_images` | `true` | Cache the shared answer image |
| `dataset.score_resolution` | `0.05` | Evaluation score grid |
| `dataset.max_criterion_score` | `4.0` | Safe fallback for malformed unseen maxima |
| `dataset.repair_criterion_max` | `true` | Enable training-only maximum repair |

### Model-specific settings

| Model | Image side | LoRA batch | LoRA accumulation | Full batch | Full accumulation | Inference batch |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL | 1280 | 4 | 4 | 2 | 8 | 8 |
| InternVL3 | 1280 | 4 | 4 | 2 | 8 | 8 |
| Pixtral | 768 | 2 | 8 | 1 | 16 | 4 |
| Donut | 1280 | 4 | 4 | 4 | 4 | 8 |

Each `models.<key>` entry also defines `family`, `model_id`, and `model_class`. Donut overrides LoRA targets with `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, and `fc2`.

### Donut

| Key | Default | Meaning |
|---|---|---|
| `donut.task_start_token` | `<s_grade>` | Decoder task token |
| `donut.image_size` | `[1280, 640]` | Height and width for the processor |
| `donut.max_text_length` | `512` | Decoder prefix budget |
| `donut.max_prompt_chars` | `1200` | Reference answer character limit |
| `donut.max_new_tokens` | `16` | Generated score token budget |

### Model loading

| Key | Default | Meaning |
|---|---|---|
| `model_loading.torch_dtype` | `bfloat16` | Weight and compute dtype |
| `model_loading.device_map` | `auto` | Transformers device placement |
| `model_loading.trust_remote_code` | `true` | Allow model repository code |
| `model_loading.attn_implementation` | `eager` | Required for attention-based XAI |
| `model_loading.load_in_4bit` | `false` | Optional 4-bit loading |
| `model_loading.load_in_8bit` | `false` | Optional 8-bit loading |

Quantized loading is not used for the `full` regime because base parameters must be updated.

### Few-shot prompting

| Key | Default | Meaning |
|---|---:|---|
| `few_shot.n_exemplars` | `3` | Requested graded examples |
| `few_shot.stratify_by_score` | `true` | Sample across score levels |
| `few_shot.exemplar_seed` | `42` | Exemplar selection seed |
| `few_shot.max_exemplar_image_side` | `512` | Exemplar image size limit |
| `few_shot.max_length` | `8192` | Few-shot prompt token budget |

If the prompt does not fit, exemplars are removed until it fits. Image placeholder sequences are never truncated.

### Training

| Key | Default | Meaning |
|---|---:|---|
| `training.epochs` | `3` | Training epochs |
| `training.learning_rate` | `1e-4` | LoRA learning rate |
| `training.warmup_ratio` | `0.05` | Warmup share |
| `training.weight_decay` | `0.01` | Weight decay |
| `training.lr_scheduler` | `cosine` | Learning-rate scheduler |
| `training.max_grad_norm` | `1.0` | Gradient clipping norm |
| `training.max_length` | `8192` | Training sequence limit |
| `training.freeze_vision_tower` | `true` | Freeze image encoder |
| `training.bf16` | `true` | Prefer bfloat16 with fallback |
| `training.gradient_checkpointing` | `true` | Reduce activation memory |
| `training.dataloader_num_workers` | `2` | Training loader workers |
| `training.optim` | `adamw_torch_fused` | Optimizer |

LoRA settings:

| Key | Default |
|---|---|
| `training.lora.r` | `32` |
| `training.lora.alpha` | `64` |
| `training.lora.dropout` | `0.05` |
| `training.lora.target_modules` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |

Full-regime settings:

| Key | Default | Meaning |
|---|---:|---|
| `training.full.learning_rate` | `2e-5` | Base-weight learning rate |
| `training.full.chat_unfreeze_fraction` | `0.30` | Target fraction for top chat-model language blocks |
| `training.full.force` | `false` | Bypass memory feasibility check |

Donut updates all parameters in the full regime. Chat VLMs freeze the vision tower, projector, embeddings, and lower language blocks, then train a suffix of complete top language blocks nearest the configured fraction.

### Inference and ensemble

| Key | Default | Meaning |
|---|---:|---|
| `inference.max_new_tokens` | `128` | Generation budget |
| `inference.do_sample` | `false` | Deterministic main evaluation |
| `inference.temperature` | `0.0` | Main evaluation temperature |
| `inference.max_length` | `4096` | Input sequence budget |
| `inference.num_workers` | `2` | Inference loader workers |
| `inference.clamp_to_criterion_max` | `true` | Clamp predictions to valid bounds |
| `ensemble.small_models` | Qwen, InternVL | First-stage models |
| `ensemble.tiebreaker` | Pixtral | Escalation model |
| `ensemble.disagree_threshold` | `1.0` | Escalation threshold in marks |
| `ensemble.clamp_to_max` | `true` | Clamp ensemble output |
| `ensemble.snap_to_grid` | `true` | Snap ensemble scores |
| `ensemble.score_resolution` | `0.25` | Ensemble output grid |

### Study settings

| Study | Key | Default | Meaning |
|---|---|---:|---|
| 1 | `study1.adjacent_tolerance` | `1.0` | Adjacent-match tolerance in marks |
| 2 | `study2.compare_runs` | `all` | Runs included in agreement analysis |
| 2 | `study2.bootstrap_ci` | `true` | Enable confidence intervals |
| 2 | `study2.n_bootstrap` | `2000` | Bootstrap draws |
| 3 | `study3.n_runs` | `5` | Repeated stochastic runs |
| 3 | `study3.do_sample` | `true` | Enable stochastic decoding |
| 3 | `study3.temperature` | `0.7` | Reliability temperature |
| 3 | `study3.top_p` | `0.9` | Reliability nucleus threshold |
| 3 | `study3.regime` | `lora` | Reliability regime |
| 4 | `study4.ink_density_quantiles` | `[0.33, 0.66]` | Handwriting density cut points |
| 4 | `study4.answer_length_quantiles` | `[0.33, 0.66]` | Ink-pixel cut points |
| 4 | `study4.large_error_threshold` | `2.0` | Large-error threshold |
| 4 | `study4.min_subgroup_n` | `10` | Minimum subgroup size |
| 4 | `study4.subgroup_features` | six configured features | Fairness dimensions |
| 5 | `study5.xai_model` | `qwen25vl` | Explanation model |
| 5 | `study5.xai_regime` | `lora` | Explanation checkpoint regime |
| 5 | `study5.n_heatmap_samples` | `12` | Number of explanation images |
| 5 | `study5.xai_max_image_side` | `768` | XAI image size limit |
| 5 | `study5.max_seq_for_rollout` | `3072` | XAI sequence safety limit |
| 5 | `study5.attribution_method` | `contrastive_score_attention` | Attribution algorithm |
| 5 | `study5.attention_last_k_layers` | `8` | Attention layers used |
| 5 | `study5.attention_baseline_queries` | `4` | Baseline query count |
| 5 | `study5.faithfulness.n_samples` | `30` | Faithfulness examples |
| 5 | `study5.faithfulness.mask_fractions` | `[0.1, 0.2, 0.3, 0.5]` | Deleted image fractions |
| 5 | `study5.faithfulness.metric` | `score_change` | Faithfulness response |
| 5 | `study5.survey.n_items` | `10` | Teacher-survey examples |
| 5 | `study5.survey.likert_max` | `5` | Maximum Likert score |

### Optional Hub uploads

| Key | Default | Meaning |
|---|---|---|
| `hub.push_adapters` | `false` | Upload LoRA adapters |
| `hub.push_results` | `false` | Upload result directories |
| `hub.models_repo` | `khalidsyfullah/obe-models` | Adapter repository |
| `hub.results_repo` | `khalidsyfullah/obe-results` | Results repository |

Authenticate with `hf auth login` before enabling either upload option.

## File reference

### Root files

| File | Purpose |
|---|---|
| `README.md` | Setup, execution, configuration, and file reference |
| `CHANGELOG.md` | Release history and important implementation fixes |
| `config.yaml` | All experiment, model, training, inference, and study hyperparameters |
| `requirements.txt` | Python dependencies for models, studies, dataset generation, and figures |
| `.gitignore` | Excludes caches, environments, checkpoints, and generated publication artifacts |

### Core package

| File | Purpose |
|---|---|
| `src/__init__.py` | Package metadata and version |
| `src/config.py` | Configuration loading, validation, seeds, and canonical paths |
| `src/data.py` | Dataset loading, image caching, features, and few-shot selection |
| `src/ensemble.py` | Cascaded ensemble routing and score combination |
| `src/hub.py` | Optional adapter and result uploads using cached authentication |
| `src/inference.py` | Batched generation, parsing, clamping, and JSON prediction output |
| `src/io_utils.py` | CSV and JSON result writers plus Study 5 PNG saving |
| `src/metrics.py` | Accuracy, agreement, kappa, ICC, reliability, and bootstrap metrics |
| `src/models.py` | Model registry, quantization, processors, LoRA, and checkpoint loading |
| `src/prompting.py` | Grading prompts and chat-template compatibility |
| `src/scoring.py` | Criterion-maximum repair and decimal score grids |
| `src/training.py` | LoRA and resource-aware full-regime training |
| `src/xai.py` | Score-conditioned attention and deletion faithfulness |

### Executable scripts

| File | Purpose |
|---|---|
| `scripts/generate_dataset.py` | Reproducible PDF-to-Hugging-Face dataset builder |
| `scripts/inspect_dataset.py` | Dataset audit and summary tables |
| `scripts/export_grading_pack.py` | Human-rating booklet, sheet, and protected data images |
| `scripts/run_experiment.py` | One model and one regime |
| `scripts/run_ensemble.py` | One cascaded ensemble regime |
| `scripts/run_all.py` | Resumable sweep and optional study runner |
| `scripts/study1_accuracy.py` | Accuracy analysis |
| `scripts/study2_human_agreement.py` | Human agreement and ceiling analysis |
| `scripts/study3_reliability.py` | Repeated-run reliability analysis |
| `scripts/study4_fairness_error.py` | Subgroup fairness and error taxonomy |
| `scripts/study5_xai_eval.py` | XAI images, faithfulness, and teacher survey analysis |

### Figure Generation Script

| File | Purpose |
|---|---|
| `Figure Generation Script/README.md` | Figure package inputs, outputs, and commands |
| `Figure Generation Script/figure_4_accuracy.py` | Figure 4 accuracy panels |
| `Figure Generation Script/figure_5_human_agreement.py` | Figure 5 agreement and human ceiling panels |
| `Figure Generation Script/figure_6_reliability.py` | Figure 6 reliability panels |
| `Figure Generation Script/figure_7_fairness.py` | Figure 7 fairness and error panels |
| `Figure Generation Script/figure_8_xai.py` | Figure 8 faithfulness and survey panels |
| `Figure Generation Script/figure_9_heatmaps.py` | Figure 9 explanation examples |

### Notebook and tests

| File | Purpose |
|---|---|
| `notebooks/OBE_Dataset_Generation.ipynb` | Colab front end for dataset generation without embedded credentials |
| `tests/test_core.py` | Core scoring, configuration, prompting, ensemble, and metric tests |
| `tests/test_outputs.py` | CSV and JSON output contract tests |
| `tests/test_cli_pipeline.py` | End-to-end CPU command-line pipeline smoke test |
| `tests/test_training_xai.py` | Training, checkpoint, inference, and XAI regression tests |

### Data and results

| Path | Contents |
|---|---|
| `data/human_ratings.csv` | Human criterion scores for Study 2 |
| `data/teacher_survey_responses.csv` | Teacher responses for Study 5 |
| `data/grading_pack/grading_booklet.html` | Browser-based human-grading interface |
| `data/grading_pack/grading_sheet.csv` | Human-grading sheet in CSV format |
| `data/grading_pack/grading_sheet.xlsx` | Human-grading sheet in XLSX format |
| `data/grading_pack/images/` | Protected answer images used by the grading pack |
| `results/runs/<model>/<regime>/predictions_run0.json` | Released prediction records as JSON arrays |
| `results/runs/<model>/<regime>/metrics.{csv,json}` | Overall run metrics |
| `results/runs/<model>/<regime>/metrics_per_criterion.{csv,json}` | Criterion-level metrics |
| `results/runs/<model>/<regime>/reliability/` | Study 3 repeated prediction JSON files |
| `results/studies/study1/` | Accuracy tables and summary JSON |
| `results/studies/study2/` | Human-agreement tables and summary JSON |
| `results/studies/study3/` | Reliability tables and summary JSON |
| `results/studies/study4/` | Fairness, taxonomy, examples, and summary JSON |
| `results/studies/study5/` | Faithfulness and survey CSV/JSON plus retained PNG explanation images |
| `results/studies/study5/teacher_survey_responses.csv` | Raw Study 5 ratings used directly by Figure 8 |
| `results/sweep_summary.{csv,json}` | Released sweep status summary |

## Reproducibility notes

- Relative paths resolve from the repository root.
- Main predictions are deterministic by default.
- Study 3 uses seed `seed + run_index` with stochastic decoding.
- Prediction files are standard JSON arrays, not JSON Lines.
- Tables are written in CSV and JSON only.
- Main experiment and Studies 1 through 4 do not generate figures.
- Publication figures are reproducible from released results with the dedicated figure scripts.
- Large model runs require suitable GPU memory. The configured full regime targets a single 96 GB GPU for each chat VLM and is not equivalent to training every chat-model parameter.

## LLM usage statement

Large language models were used to organize the codebase, improve documentation, and assist with refactoring. The research design, data, results, and scientific conclusions remain the authors' responsibility.
