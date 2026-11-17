# LegibleRx — Predicting Low-Vision Legibility of Medicine-Strip Text

ES 678 course project. Given a photo of a medicine strip or carton and its text-region boxes, the model predicts a
**legibility severity score (0–8)** for each region: the number of steps on a simulated low-vision ladder (blur,
contrast loss, veiling glare) the text survives before OCR fails (CER ≥ 10 %). The score is also binned into
illegible / borderline / legible.

## Repository layout
| Path | What it is |
|---|---|
| `LegibleRx_colab.ipynb` | Full pipeline: data, proxy labels, features, split, models, evaluation, depth analyses |
| `legiblerx_pipeline.py` | The same code as a plain script (`python legiblerx_pipeline.py`) |
| `results/` | Outputs of the executed run: metrics CSV/JSON, predictions, figures |
| `paper/` | ES 678 paper: `legiblerx_paper.tex`, `legiblerx_paper.pdf`, figures, `neurips_2026.sty` |
| `synthetic_data/` | The 300 synthetic pack images and `regions.csv` used in the reported run (regenerable from the seed) |

## Main result (synthetic data, 20 held-out products, 360 regions)
| Model | MAE | Spearman | macro-F1 [95% CI] | QWK | Illegible recall |
|---|---|---|---|---|---|
| Rule baseline | 1.63 | 0.64 | 0.59 [0.53, 0.66] | 0.60 | 0.86 |
| LightGBM | 0.81 | 0.85 | 0.81 [0.75, 0.86] | 0.77 | 0.70 |
| CNN (random init) | 1.63 | 0.68 | 0.69 [0.62, 0.75] | 0.64 | 0.58 |

The notebook in this repo is already executed, so all outputs are visible without re-running. The reported run used
100 synthetic products and a CPU (the CNN had no ImageNet weights because they could not be downloaded there).

## Reproduce
1. Open `LegibleRx_colab.ipynb` in Google Colab → *Runtime → Change runtime type → T4 GPU*.
2. In the **Configuration** cell set `DATA_MODE`:
   - `"synthetic"` (default) — generates 100 synthetic packs × 3 lightings; no download needed.
   - `"real"` — point `DATA_DIR` at the dataset folder (format below).
3. *Runtime → Run all*. About 20–30 min on Colab (most of it is the OCR ladder). `results.zip` downloads at the end.

Seeds are fixed (`SEED = 42`). Local run: `apt install tesseract-ocr && pip install -r requirements.txt`, then
`python legiblerx_pipeline.py` (`QUICK=1` for a 5-minute smoke test).

Tested with Python 3.12, scikit-learn 1.x, LightGBM 4.x, SHAP 0.4x, PyTorch 2.x, Tesseract 5.

## Data access
* **Synthetic mode** needs no data and is fully regenerated from the seed.
* **Real photos** of medicine packs (front/back of strips and cartons) were taken by the team. No patient data and no
  personal information are included. *[Say here whether the photos are in this repo, in a linked Drive folder,
  or available on request, and include a small permitted sample in `sample_data/`.]*

Expected layout for `DATA_MODE="real"`:
```
DATA_DIR/images/<image_file>
DATA_DIR/regions.csv        image_file, product_id, lighting, region_id, x1, y1, x2, y2, text, region_type
DATA_DIR/human_ratings.csv  (optional) region_id, rater_id, rating      # 1–5, held out from training
DATA_DIR/expert_tags.csv    (optional) region_id, cause                  # size|contrast|glare|curvature|weight|clutter
```

## Method summary (what to check)
* **Split:** by `product_id` (GroupShuffleSplit, 20 % of products held out as test). No pack appears in both.
* **Tuning:** LightGBM hyper-parameters and class cut-offs chosen by 5-fold GroupKFold on the dev set only;
  CNN early-stopped on a product-disjoint validation set. Test is scored once.
* **Leakage:** features are deterministic per-region measurements; the only fitted transform (feature
  standardisation for the CNN) is fit on the training split.
* **Baselines:** dummy (mean / majority) and a print-guideline rule (x-height, WCAG 4.5:1 contrast, glare).
* **Metrics:** MAE, RMSE, R², Spearman ρ, macro-F1, balanced accuracy, quadratic-weighted κ, illegible recall,
  safety-critical subset (drug name, strength, expiry), product-clustered bootstrap 95 % CIs, paired ΔF1 vs rule.
* **Depth analyses:** SHAP, feature-group ablation, τ-sensitivity of the proxy label, split-conformal 90 %
  intervals, failure audit by lighting and region type, latency.

## Limitations
The label is an OCR proxy for human legibility; it depends on the OCR engine (Tesseract) and on the simulator
parameters. Results from synthetic mode do not show real-world performance. Region boxes and types are given as
input; detection errors are not evaluated.

## AI-use disclosure
*[Edit to match what you actually did.]* The pipeline code was drafted with the help of Claude (Anthropic), an AI
assistant, from our project proposal. We reviewed, ran and modified the code, and we chose the evaluation design and
wrote the analysis and paper ourselves.

## Citations and external code
* Tesseract OCR (Apache-2.0) via `pytesseract`; LightGBM (MIT); SHAP (MIT) — Lundberg & Lee, *NeurIPS* 2017;
  PyTorch/torchvision MobileNetV3 — Howard et al., *ICCV* 2019; scikit-learn — Pedregosa et al., *JMLR* 2011.
* Split-conformal prediction — Lei et al., *JASA* 2018.
* WCAG 2.2 contrast-ratio definition, W3C.
* Latham et al., 2011 (pharmacy-label legibility in visual impairment); Univ. of Waterloo & CNIB prototype labels;
  Connors (container shape and label readability); Caballero Sandoval, 2024; Jiang & Xu, 2024 (MedReadMe).
* Fonts used to render synthetic packs: Liberation Sans / DejaVu Sans (open licences).

No credentials, API keys or private data are stored in this repository.
