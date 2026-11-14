# AI-Based Multimodal Sports Performance Assessment System

Upload a cricket or badminton video. The system finds the player's body keypoints, recognises the
shot, measures the technique, scores it out of 100 and gives coaching feedback on a web dashboard.

## What is included (and what is not)

| Part | Status |
|---|---|
| Pose extraction (MediaPipe, optional YOLO tracking), features, GRU action model, ML scorer, feedback engine, Streamlit dashboard, SQLite history, evaluation metrics | Written |
| Training data (videos) and score labels | **You must collect these.** No code can replace them. |
| Trained models | **You train them** (about 1 hour on Colab once data is ready). |

Tested here on fake data: features, scoring, feedback, database. **Not tested here:** the PyTorch
action model, real video pose extraction and the Streamlit app (these need torch, mediapipe 0.10.14 and a real video).
Run `python tests/smoke_test.py` on your machine first; it checks everything on fake data.

## How it works

```
Video -> frames -> (optional YOLO tracking) -> MediaPipe pose (33 keypoints)
   -> GRU action recogniser  : shot type + confidence        (Deep Learning)
   -> biomechanical features : angles, balance, rotation...  (numerical modality)
   -> motion energy          : frame differences             (video modality)
   -> fusion: features + motion + action probabilities + GRU embedding
   -> RandomForest / GradientBoosting / XGBoost : scores     (Machine Learning)
   -> rule-based feedback -> Streamlit dashboard
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
python tests/smoke_test.py       # should end with ALL CHECKS PASSED
```
On Colab, use `pip install mediapipe==0.10.14` (newer versions removed the API this project uses).

## Step-by-step use (cricket example)

1. **Collect videos** into `data/raw_videos/cricket/<action>/`, for example
   `cover_drive`, `pull_shot`, `defensive_shot`. Use one fixed camera view. Aim for 50+ clips per action,
   each clip trimmed to one shot. Check the licence of any public dataset you use.
2. **Extract keypoints:** `python scripts/extract_dataset.py --sport cricket`
3. **Train the action model:** `python scripts/train_action.py --sport cricket`
   (prints accuracy, precision, recall, F1; saves the confusion matrix in `models/cricket/`).
4. **Create the label sheet:** `python scripts/make_label_template.py --sport cricket`
   Then have a coach or experienced player rate each clip 0-100 for every parameter in
   `data/labels/cricket_scores.csv`. Use 2 raters if you can and average them. Fill `player_id` if you know it.
   Label at least 100-300 clips.
5. **Train the scoring model:** `python scripts/train_scoring.py --sport cricket`
   (compares RandomForest, GradientBoosting and XGBoost with cross-validation: MAE, RMSE, R2).
6. **Test on one video:** `python scripts/analyze_video.py --sport cricket --video clip.mp4`
7. **Run the dashboard:** `streamlit run app/streamlit_app.py`
8. **Badminton:** repeat steps 1-7 with `--sport badminton`. Nothing in the code changes.

## Adding a new sport
Add one entry in `src/config.py` (`SPORTS`) with its actions and score parameters. Put videos in
`data/raw_videos/<sport>/<action>/` and follow the same steps. Add advice text for new parameters in `src/feedback.py`.

## Folder map
```
src/config.py        sports, actions, parameters, paths
src/pose.py          video -> keypoints, tracking, skeleton drawing
src/features.py      angles, balance, rotation... and the sequence for the GRU
src/action_model.py  GRU classifier, training, evaluation, inference
src/scoring.py       fusion + ML scorer + model comparison
src/feedback.py      rule-based coaching text
src/pipeline.py      full video -> result
src/db.py            SQLite history
app/streamlit_app.py dashboard
scripts/             extract, label template, train, analyze
tests/               smoke test on fake data
```

## Limitations to state in your report
- Keypoints come from MediaPipe; they are less accurate with blur, occlusion or low resolution.
- Joint angles depend on the camera view; train and test with the same view, or label several views separately.
- The scorer can only be as good as the human labels. Report how many raters, and how much they agreed.
- Small datasets overfit. Split by player (use `player_id`) so the same person is not in training and testing.
- Score labels are subjective; the model learns the raters' opinion, not an absolute truth.
- "Multimodal" here means fusing video motion, pose sequence, and numerical features. All come from one video; say this openly.
- Speed features are per frame; keep the same frame rate across clips.

## Suggested report outline
1 Introduction and objectives. 2 Literature review (pose estimation, action recognition). 3 System architecture (diagram above).
4 Dataset and labelling method. 5 Methodology (pose, features, GRU, scorer, fusion). 6 Implementation (Streamlit, SQLite).
7 Results: `models/<sport>/action_metrics.json`, `confusion_matrix.png`, `scoring_comparison.csv`, top features in `scoring_metrics.json`.
8 Limitations. 9 Future scope (more sports, real-time, wearable sensors, LLM feedback).
