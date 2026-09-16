# How Linguistic, Visual, and Audio Styles Perform Populism

Code and data for the study *"How Linguistic, Visual, and Audio Styles Perform Populism: A Multimodal, Computational Analysis of Campaign Videos on Instagram."* This repository contains the full processing pipeline — from Instagram data collection through audio/visual/text feature extraction, populism classification, clustering, and the final statistical models — plus the analysis-ready datasets.

> **Citation**: *TODO — fill in the full citation (authors, year, journal, DOI) once the paper is accepted/published.*

## Getting the code

```bash
git clone <this-repo-url>
cd "<this-repo-folder>"
```

This repository is also archived on OSF: *TODO — add the OSF project link here once published.* The two are kept in sync with identical contents; use whichever is more convenient (`git clone` here for the latest version and version history, or download directly from OSF for a one-off copy without needing git).

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── code/
│   ├── data crawling/
│   │   ├── import_browser_session.py
│   │   ├── ig_get_video.py
│   │   ├── ig_scrape_post.py
│   │   └── ig_scrape_comment.py
│   ├── shared/
│   │   ├── feature_extraction.ipynb
│   │   └── kmeans_clustering.ipynb
│   ├── audio processing/
│   │   ├── data_denoise.py
│   │   ├── asv_voice_detection.py
│   │   ├── get_data_per_sec.py
│   │   ├── get_emotion.py
│   │   ├── get_emotion_mean_std.py
│   │   ├── get_speaking_rate.py
│   │   ├── hierarchical_2024_0901_1130.py
│   │   ├── kmeans.ipynb
│   │   ├── kmeans_2024_0901_1130.py
│   │   ├── plot_with_emotions.py
│   │   └── prepare_data_2024_0911_1130.py
│   ├── follower recalibration/
│   │   └── engagement_regression_predict.py
│   ├── linguistic feature selection/
│   │   └── LIWC_features.py
│   ├── popBert/
│   │   ├── popBERT1.py
│   │   ├── LIWC-22 Results - labeled_populism_v38_N398 - LIWC Analysis.csv
│   │   ├── train_set.csv
│   │   ├── test_set.csv
│   │   └── final_metrics.csv
│   ├── portrait generation/
│   │   ├── audio_generation.ipynb
│   │   └── visual_generation.ipynb
│   ├── text emotion/
│   │   └── model_load_emo.py
│   ├── text processing/
│   │   ├── check_best_by_nmi.ipynb
│   │   ├── k_means_bertopic_pipline.ipynb
│   │   └── topic_prompt.ipynb
│   └── video processing/
│       ├── preprocess_recognition_v2.py
│       ├── pyfeat_video_multiprocessing.py
│       ├── visual_valence_arousal_loop.py
│       ├── feature_extract_valenceArousal.py
│       └── emonet/            # vendored EmoNet model code (Toisoul et al. 2021)
├── dataset/
│   ├── audio_cluster_54_k5_v1.csv
│   ├── combined_240901-241104_N583_20251001.csv
│   ├── combined_241001-241104_N398_caption_post_populism.csv
│   ├── data_N398_250909.csv
│   ├── features_raw_251001.csv
│   ├── ig_video_N398_post,comment,whisper_cluster_sentiment_statistics(before0109)_0716_v3.csv
│   ├── ig_video_visualValid_N583_0901-1104_zscoreEngagement_250909.csv
│   ├── ig_video_visualValid_N584_0901-1104_zscoreEngagement_0328.csv
│   ├── labeled_text_emo_N9675_sentenced_v5.csv
│   ├── labeled_populism_v38_N398.csv
│   ├── labeled_populism_v38_N398 - LIWC Analysis.csv
│   ├── Poll_NYTimes_241001-241104_v2.csv
│   ├── text_cluster_33_k4_v_auto.csv
│   ├── visual_audio_cluster_54_k5_v1.csv
│   ├── visual_cluster_54_k6_v1.csv
│   └── word_data_398.csv
└── statistics/
    └── stats_N398_1002.rmd
```

All data files that more than one script reads (`audio_cluster_54_k5_v1.csv`, `features_raw_251001.csv`, `combined_241001-241104_N398_caption_post_populism.csv`, `labeled_populism_v38_N398 - LIWC Analysis.csv`) live only in `dataset/` — the scripts and notebooks that use them (`LIWC_features.py`, `visual_generation.ipynb`, `audio_generation.ipynb`, `stats_N398_1002.rmd`) read them from there via a relative path or an environment-variable override. `stats_N398_1002.rmd`'s remaining inputs are all in `dataset/` too, except 3 that are read but not actually used (see the comments in its first code chunk) and can be skipped. `Poll_NYTimes_241001-241104_v2.csv` is third-party NYTimes polling data — confirm you have the right to redistribute it before publishing this repository publicly.

**Verified reproducible**: with every file above in place, the full merge chain in `stats_N398_1002.rmd`'s first code chunk (`combined1` → `combined2` → `data`) has been checked to preserve all 398 rows through every join, with zero missing values in the resulting cluster/topic/populism columns, and the visual/audio/linguistic cluster sizes match the paper's reported tables exactly (Table 21/23/25). One bug was fixed to get there: the chunk originally read `labeled_populism_v38_N398 - LIWC Analysis.csv` for the `populism` object, but that file only has LIWC category columns — the `centr`/`elite`/`left`/`right`/`party`/`keycode` columns the code actually needs (matching `popBERT1.py`'s `label_cols`) live in `labeled_populism_v38_N398.csv`, which is what the chunk now reads.

## Environment setup

**Python** (used by everything under `code/` except `statistics/`):

```bash
pip install -r requirements.txt
```

`requirements.txt` is compiled from the imports actually used in this repo and is intentionally unpinned — see the comment at the top of that file for how to produce a fully pinned lockfile. A few packages need extra setup:

- `torch`: install the CUDA build matching your GPU first via https://pytorch.org/get-started/locally/, before installing the rest of `requirements.txt`.
- `face_recognition`: requires `dlib`, which needs CMake and a C++ compiler to build.
- `liwc` (used by `LIWC_features.py`): only provides the parser — you must separately obtain `LIWC2022_English.dic` from https://www.liwc.app/ (proprietary, not redistributable here) and point `LIWC_DIC_PATH` at it.

**R** (used by `statistics/stats_N398_1002.rmd`):

```r
install.packages(c(
  "tidyverse", "lme4", "glmmTMB", "ggeffects", "mediation", "vcd",
  "ggpattern", "openxlsx", "ggpubr", "ggsignif", "fixest", "gridExtra", "readr", "stringr"
))
```

**Secrets and machine-specific paths**: scripts read their inputs/outputs and API credentials from environment variables (e.g. `OPENAI_API_KEY`, `HF_TOKEN`, `DATA_ROOT`, `OUTPUT_ROOT`) rather than hardcoded values — each script's `__main__` block or top-of-file config shows which variables it expects and what relative-path default it falls back to. Set these in your shell (or a local `.env` you `source` yourself) before running; never commit real credentials — check `git status` before every commit, especially after running `data crawling/import_browser_session.py` (which writes a cookie/session file to disk) or downloading the EmoNet/GoEmotion model weights described above.

## Execution order

The `code/` subfolders are not independent — later stages consume CSVs produced by earlier ones. Rough pipeline order:

1. **`data crawling/`** — `import_browser_session.py` (creates the login session) → `ig_get_video.py` (downloads video posts and basic metadata, including raw `post_likes`/`post_comments` counts, for a list of candidate Instagram accounts within a date range) → later, once the video/engagement pipeline below has filtered the dataset down to valid posts, `ig_scrape_post.py` / `ig_scrape_comment.py` (re-scrape captions and comments for that filtered post list). See the "Data crawling walkthrough" section below for full step-by-step instructions, and each script's docstring for a Terms-of-Service caveat.
2. **`video processing/`** — `preprocess_recognition_v2.py` (MediaPipe face detection + face-recognition matching to the target candidate) → `pyfeat_video_multiprocessing.py` (py-feat facial action-unit and head-pose extraction) → `visual_valence_arousal_loop.py` (EmoNet valence/arousal/discrete-emotion inference per frame, using the vendored `emonet/` model code; requires pretrained weights not included in this repo, see its docstring) → `feature_extract_valenceArousal.py` (aggregates EmoNet's per-frame output to per-video mean/std).
3. **`audio processing/`** — `data_denoise.py` (Demucs vocal separation) → `get_data_per_sec.py` (pyannote speaker diarization) → `asv_voice_detection.py` (ReDimNet speaker-embedding cosine-similarity matching against reference candidate recordings) → `get_speaking_rate.py` (Whisper transcription + speaking rate) → `get_emotion.py` / `get_emotion_mean_std.py` (wav2vec2 valence/arousal/dominance) → `prepare_data_2024_0911_1130.py`, `hierarchical_2024_0901_1130.py`, `kmeans_2024_0901_1130.py`, `kmeans.ipynb`, `plot_with_emotions.py` (exploratory clustering and plots).
   - Note: pitch and vowel formants (Praat via Parselmouth) and duration/voiced-unvoiced ratio (OpenSMILE) are described in the paper's Online Appendix I, but no extraction script for either was recoverable — only the resulting feature CSVs are consumed by later steps. MFCC0 (SpeechBrain) is in the same situation, except a pinned environment snapshot found elsewhere in the team's records confirms `opensmile==2.5.0` and `speechbrain==1.0.2` were the versions actually used (see `requirements.txt`) — useful if you rewrite these extractors yourself. See the Online Appendix for citations to the tools used.
4. **`linguistic feature selection/`** — `LIWC_features.py` (requires the licensed LIWC-22 dictionary, see Environment setup above).
5. **`popBert/`** — `popBERT1.py` (RoBERTa-Large multilabel populism classifier: training, grid search, evaluation).
6. **`text emotion/`** — `model_load_emo.py` (fine-tuned RoBERTa-large 7-class GoEmotion-style classifier applied to sentence-level transcripts; requires the fine-tuned model checkpoint, ~1.4GB, not included in this repo — see its docstring).
7. **`text processing/`** — `k_means_bertopic_pipline.ipynb` (BERTopic topic modeling on transcripts) → `check_best_by_nmi.ipynb` (cluster-count selection) → `topic_prompt.ipynb` (GPT-4o topic labeling; requires `OPENAI_API_KEY`).
8. **`shared/`** — `feature_extraction.ipynb` (merges the modality-specific feature CSVs from steps 2–6 into per-video tables) → `kmeans_clustering.ipynb` (k-means clustering per modality, selects k via silhouette score).
9. **`portrait generation/`** — `visual_generation.ipynb` / `audio_generation.ipynb` (illustrative GPT-image/TTS cluster portraits; optional, for visualization only; requires `OPENAI_API_KEY`).
10. **`follower recalibration/`** — `engagement_regression_predict.py` (Instagram follower-count backfill/calibration described in the paper's Online Appendix B).
11. **`statistics/`** — `stats_N398_1002.rmd` (final GLMM and Tweedie regression models, run last on the merged dataset).

## Data crawling walkthrough

Step-by-step instructions for `code/data crawling/`, from a fresh checkout to a downloaded video dataset. Commands below use bash `export VAR=value`; on Windows PowerShell use `$env:VAR="value"` instead.

**0. Prerequisites**

- `pip install -r requirements.txt` (installs `instaloader`, `emoji`, `openpyxl` among others).
- An Instagram account to scrape with. Use a disposable/secondary account, not your personal one — this account risks being rate-limited or banned (see the Terms-of-Service note below).
- A browser (Edge, Chrome, or Firefox all work) with a cookie-export extension installed, e.g. "Cookie-Editor" (available in each browser's extension store).

**1. Log into Instagram and export the session cookie**

1. In your browser, go to instagram.com and log in with the scraping account from step 0.
2. Click the Cookie-Editor extension icon while on instagram.com.
3. Choose "Export" → "Export as JSON" (or "Copy all cookies as JSON" — either way you want the raw JSON array).
4. Save that JSON as a file named `edge_cookie.json` inside `code/data crawling/` (or anywhere else — just point `COOKIE_JSON_PATH` at it in the next step). The file should look like `[{"name": "sessionid", "value": "...", ...}, {"name": "csrftoken", "value": "...", ...}, ...]`.

**2. Turn the cookie export into an instaloader session**

```bash
cd "code/data crawling"
export COOKIE_JSON_PATH=./edge_cookie.json
export SESSION_OUTPUT_DIR=.
python import_browser_session.py
```

On success this prints `Session saved for <username> in . -- set IG_SESSION_FILE to that path for the other scripts.` and creates a file named `session-<username>` in the current directory. If you instead see `Not logged in...`, the cookie export is stale or incomplete — repeat step 1 with a fresh export.

**3. Prepare the candidate-account spreadsheet**

`ig_get_video.py` expects an Excel file (default path `./ig_get_videos_with_ID.xlsx`, override with `ACCOUNTS_XLSX`) with one row per account and these columns:

| Column | Meaning |
|---|---|
| `States` | the account's associated state (or any grouping label you use downstream) |
| `source` | a human-readable name for the account (used to name output files) |
| `Party` | party/affiliation label, if relevant to your use case |
| `name_id` | the account's Instagram profile URL, e.g. `https://www.instagram.com/username/` |
| `ID` | leave blank on the first run — the script resolves each account's numeric Instagram user ID from `name_id` and writes it back into this column so subsequent runs don't need to re-resolve it |

**4. Download videos**

```bash
export IG_ACCOUNT_USERNAME=<the username you logged in as in step 1>
export IG_SESSION_FILE=./session-<that username>
export ACCOUNTS_XLSX=./ig_get_videos_with_ID.xlsx
export OUTPUT_DIR=./data/ig_videos_raw
export START_DATE=2024-09-01   # optional, this is the default
export END_DATE=2024-11-30     # optional, this is the default
python ig_get_video.py
```

This downloads every video post published in `[START_DATE, END_DATE]` for each account in the spreadsheet, saving the `.mp4` files plus a per-account `{source}_posts.csv` (with `post_id`/`post_date`/`post_likes`/`post_comments`/`post_text`) into `OUTPUT_DIR`. It sleeps a randomized few seconds between posts and 15-30s between accounts to stay under rate limits, and every 30 accounts it reloads the session and rests 30-70s. Progress and skipped/failed accounts are appended to `log.txt` / `error_log.txt` in the working directory, so the run is safe to stop and restart (it re-processes from the account list each time, but existing per-account CSVs are simply overwritten, so remove or back up ones you want to keep before re-running). Expect this step to take hours for a full account list — it is intentionally slow to avoid triggering Instagram's abuse detection.

**5. (Later, after visual/engagement filtering) Backfill captions and comments**

Once the video-processing and engagement-scoring steps elsewhere in this pipeline have produced a filtered video dataset (a CSV with `keycode`/`party`/`state`/`name`/`post_date` columns — `dataset/ig_video_visualValid_N584_0901-1104_zscoreEngagement_0328.csv` is the shipped example), re-scrape captions and comments for exactly that filtered post list:

```bash
export IG_ACCOUNT_USERNAME=<username>
export IG_SESSION_FILE=./session-<username>
export SOURCE_CSV=../../dataset/ig_video_visualValid_N584_0901-1104_zscoreEngagement_0328.csv
export OUTPUT_CSV=./output/ig_video_posts.csv
python ig_scrape_post.py

export OUTPUT_CSV=./output/ig_video_comments.csv
python ig_scrape_comment.py
```

Both scripts are resumable: if `OUTPUT_CSV` already exists, posts whose `keycode` is already in it are skipped, so re-running after an interruption (or a session expiring mid-run) picks up where it left off.

**Troubleshooting**

- `SystemExit: Cookie import failed...` from `ig_scrape_post.py`/`ig_scrape_comment.py`, or the scripts logging errors on every post: the session has expired — repeat steps 1-2 to refresh it.
- Getting rate-limited or seeing many `Error:` lines in the console: increase the sleep durations in the script (search for `randint(` calls) and/or reduce how many accounts/posts you process per run using `ig_get_video.py`'s `range_s`/`range_e` arguments in its `main()` call.
- Remember: never commit `edge_cookie.json` or any `session-*` file — both are equivalent to a live login for the scraping account.

## Folder-level description

### code/data crawling/

- import_browser_session.py: log into Instagram in a browser (Edge was used for this project) → export that session's cookies to JSON with an extension like "Cookie-Editor", overwriting `edge_cookie.json` → run this script to turn that cookie export into the `instaloader` session file the other three scripts load — this is how you log in without ever putting a plaintext password in code. Instagram sessions expire, so repeat this whole cycle (re-login → re-export → re-run) whenever the other scripts start failing to authenticate. **Never commit the cookie export or the session file it produces**; both are equivalent to a live login for the scraping account — check `git status` before committing anything in this folder.
- ig_get_video.py: downloads video posts (and the .mp4 files themselves) published within a date range for a spreadsheet of candidate Instagram accounts, including the raw `post_likes`/`post_comments` counts for each post. Fixes an inherited bug where the periodic session-reload branch referenced an undefined `account` variable.
- ig_scrape_post.py: re-scrapes the caption for each post (`keycode`) in an already-filtered video dataset; resumable, skips posts already present in the output CSV.
- ig_scrape_comment.py: scrapes all comments for each post (`keycode`) in an already-filtered video dataset; resumable, skips posts already present in the output CSV.
- The latter three require `IG_ACCOUNT_USERNAME` and `IG_SESSION_FILE` (the session file `import_browser_session.py` produces) — see each script's docstring. Scraping Instagram this way is against its Terms of Service; included here for research transparency about how the paper's raw dataset was collected, not as an endorsement — rate limits and account bans are real risks.

### code/shared/

- feature_extraction.ipynb: builds multimodal feature tables used across audio, visual, and text analysis workflows.
- kmeans_clustering.ipynb: runs clustering experiments shared by modality-specific analysis pipelines.

### code/audio processing/

- data_denoise.py: extracts audio from video files and performs denoising with Demucs-style preprocessing.
- asv_voice_detection.py: automatic speaker verification — matches each diarized speaker segment against reference recordings of the target candidate using ReDimNet speaker embeddings and cosine similarity (loads the model via `torch.hub`, no local weights needed).
- get_data_per_sec.py: uses speaker diarization to segment audio by second and generate per-second speaker activity tables.
- get_emotion.py: loads audio files and applies a pretrained speech emotion model to estimate arousal, dominance, and valence.
- get_emotion_mean_std.py: computes mean and standard deviation of emotion-related features for downstream analysis.
- get_speaking_rate.py: estimates speaking rate from audio files.
- hierarchical_2024_0901_1130.py: performs hierarchical clustering for audio-related features.
- kmeans.ipynb: supports k-means clustering on audio-related features.
- kmeans_2024_0901_1130.py: performs k-means clustering with a specific timestamped dataset version.
- plot_with_emotions.py: creates plots that combine emotional features with clustering or audio characteristics.
- prepare_data_2024_0911_1130.py: prepares working datasets for the 2024-09-11 to 2024-11-30 analysis period.

### code/follower recalibration/

- engagement_regression_predict.py: performs engagement-related regression modeling and prediction.

### code/linguistic feature selection/

- LIWC_features.py: extracts LIWC-style linguistic features for text analysis; reads `dataset/combined_241001-241104_N398_caption_post_populism.csv` by default.

### code/popBert/

- popBERT1.py: trains or evaluates a RoBERTa-based multilabel model for populism classification.
- LIWC-22 Results - labeled_populism_v38_N398 - LIWC Analysis.csv: provides LIWC-22 results for the labeled dataset.
- train_set.csv: provides the training split for populism classification.
- test_set.csv: provides the testing split for populism classification.
- final_metrics.csv: reports the final evaluation metrics from the trained model.

### code/portrait generation/

- audio_generation.ipynb: supports audio generation and audio-based feature generation; reads `dataset/features_raw_251001.csv` and `dataset/audio_cluster_54_k5_v1.csv` by default.
- visual_generation.ipynb: supports visual generation and visual feature generation; reads the same two files.

### code/text emotion/

- model_load_emo.py: applies a fine-tuned RoBERTa-large 7-class GoEmotion-style classifier to sentence-level transcripts, producing the `anger_t`/`disgust_t`/`fear_t`/`joy_t`/`neutral_t`/`sadness_t`/`surprise_t` probability columns (plus `Pred_t`) that `shared/feature_extraction.ipynb` consumes. **The fine-tuned model checkpoint (~1.4GB) is not included in this repository** — too large for a normal git repo (Git LFS or external hosting, e.g. Hugging Face Hub, would be needed to publish it); set `GOEMOTION_MODEL_DIR` to wherever you place it. The already-labeled output (`dataset/labeled_text_emo_N9675_sentenced_v5.csv`) is included, so `feature_extraction.ipynb` works without needing to install or run this script yourself.

### code/text processing/

- check_best_by_nmi.ipynb: evaluates clustering quality by comparing different NMI-based selection criteria.
- k_means_bertopic_pipline.ipynb: runs a topic modeling pipeline based on BERTopic and k-means clustering.
- topic_prompt.ipynb: supports prompt-based topic analysis or topic generation workflows.

### code/video processing/

- preprocess_recognition_v2.py: preprocesses visual recognition or video-based feature extraction data.
- pyfeat_video_multiprocessing.py: performs video feature extraction or recognition tasks with multiprocessing.
- visual_valence_arousal_loop.py: runs face detection (SFD, via `face-alignment`) and EmoNet on each video frame to produce a `*_va.csv` of per-frame time/emotion/valence/arousal. Needs pretrained EmoNet weights (not included — see its docstring for where to get them) placed in a `pretrained/` folder next to the script.
- feature_extract_valenceArousal.py: aggregates the `*_va.csv` files from `visual_valence_arousal_loop.py` into per-video emotion-distribution and valence/arousal mean/std statistics.
- emonet/: vendored copy of the EmoNet model architecture and training utilities (Kossaifi, Toisoul & Bulat; Toisoul et al. 2021, https://github.com/face-analysis/emonet) that `visual_valence_arousal_loop.py` imports from. Upstream license: **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**, non-commercial use only. `visual_valence_arousal_loop.py` itself is an adapted (modified) version of the upstream `demo_video.py` — the ND clause technically restricts distributing modified versions of their code, so if you publish this repository publicly, keep the attribution in that script's docstring intact and be aware this specific file carries that licensing risk (the unmodified files directly under `emonet/` do not, since they are shared as-is with attribution).

### dataset/

- audio_cluster_54_k5_v1.csv: audio clustering results (5 clusters); consumed by `stats_N398_1002.rmd` and the `portrait generation/` notebooks.
- combined_240901-241104_N583_20251001.csv: combines campaign video data from the 2024-09-01 to 2024-11-04 period.
- combined_241001-241104_N398_caption_post_populism.csv: combined captions, posts, and populism-related labels; consumed by `LIWC_features.py` and `stats_N398_1002.rmd`.
- data_N398_250909.csv: provides the subset of data used for the N=398 analysis sample.
- features_raw_251001.csv: raw multimodal feature table; consumed by `stats_N398_1002.rmd` and the `portrait generation/` notebooks.
- ig_video_N398_post,comment,whisper_cluster_sentiment_statistics(before0109)_0716_v3.csv: per-video post/comment/transcript sentiment scores; consumed by `stats_N398_1002.rmd`.
- ig_video_visualValid_N583_0901-1104_zscoreEngagement_250909.csv: engagement/follower z-scores for the N=583 (pre-visual-filtering) sample; the engagement table `stats_N398_1002.rmd` actually reads.
- ig_video_visualValid_N584_0901-1104_zscoreEngagement_0328.csv: a different engagement snapshot (N584, calibrated 0328) — not the one `stats_N398_1002.rmd` reads; kept for reference, see the comment there before treating the two as interchangeable.
- labeled_text_emo_N9675_sentenced_v5.csv: sentence-level GoEmotion-style output (7 emotion probabilities + top label) produced by `code/text emotion/model_load_emo.py`; consumed by `shared/feature_extraction.ipynb`.
- labeled_populism_v38_N398.csv: per-video PopBERT populism scores (`elite`/`centr`/`left`/`right`, matching `popBERT1.py`'s `label_cols`) plus `party`/`state`/`keycode`; consumed by `stats_N398_1002.rmd` for the `populism`/`populism1` objects.
- labeled_populism_v38_N398 - LIWC Analysis.csv: pure LIWC-22 category scores for the same videos (no populism labels); consumed by `stats_N398_1002.rmd` for the separate `liwc` transposed-feature table further down.
- Poll_NYTimes_241001-241104_v2.csv: NYTimes state-level polling data used as a covariate; consumed by `stats_N398_1002.rmd`. Third-party data — confirm redistribution rights before publishing.
- text_cluster_33_k4_v_auto.csv: linguistic-style clustering results (4 clusters: authentic/polite/anxious/combative); consumed by `stats_N398_1002.rmd`.
- visual_audio_cluster_54_k5_v1.csv: the base per-video table (party/state/name/win/gender/date/poll/keycode, etc.) that `stats_N398_1002.rmd` builds `data` from.
- visual_cluster_54_k6_v1.csv: visual-style clustering results (6 clusters); consumed by `stats_N398_1002.rmd`.
- word_data_398.csv: per-video topic-cluster labels; consumed by `stats_N398_1002.rmd`.

### statistics/

- stats_N398_1002.rmd: performs descriptive statistics, clustering summaries, and regression or mediation analysis.
