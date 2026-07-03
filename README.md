# 🏀 Basketball Video Analysis + AI Insight Layer

> **Fork note:** This is a fork of [abdullahtarek/basketball_analysis](https://github.com/abdullahtarek/basketball_analysis). The original detection/tracking/tactical-view pipeline and tutorial are by **Abdullah Tarek** — full credit to him for the baseline. Everything below "What I added" is mine; the original documentation follows unchanged further down.

## 🧠 What I added: turning tracking data into answers

The pipeline computes rich per-frame data (positions, speed, possession, passes) but the only outputs were an annotated video and (previously) nothing else — none of it was something a coach or player could actually *use* without translating raw numbers into meaning themselves. This closes that gap:

- **`game_report.py`** — aggregates the pipeline's outputs into one JSON report: per-player total distance + avg/max speed, per-team ball-possession %, pass/interception counts. Player-to-team attribution uses a majority vote across all frames a player appears in (not a naive per-frame lookup — `TeamAssigner` resets its mapping every 50 frames, so a single-frame lookup can be wrong). New `--report <path>` flag.
- **`llm_client.py`** — a provider-agnostic LLM client (OpenAI, Gemini, DeepSeek, or **OpenRouter** — all reachable through the identical `OpenAI(api_key=..., base_url=...)` SDK shape). Config-driven (`LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL` env vars), not hardcoded to one vendor. **OpenRouter is the recommended provider for actually trying this without paying anything**: no card required, and `openai/gpt-oss-20b:free` was confirmed via a live query of OpenRouter's own `/api/v1/models` endpoint (not just docs) to support both structured outputs and tool calling together on the free tier — unlike Gemini's OpenAI-compat layer (documented "beta", with a confirmed bug on its 2.0-series models specifically) or Groq (whose docs explicitly disallow combining strict JSON schema with tool calling in one request).
- **`game_qa.py`** — two AI features built on the report:
  - **Natural-language game summary** — one LLM call, the full report JSON in context, producing a readable recap. This is a well-established pattern (automated sports journalism — Automated Insights' AP partnership, Stats Perform's AI Studio), applied here to video-derived tracking data instead of a box score.
  - **Tool-calling Q&A chat** — ask free-form questions about the game. LLMs are measurably unreliable doing arithmetic/ranking by reading raw JSON in-context (~12% accuracy gap vs. delegating to real code, per the Program-of-Thoughts paper, [arXiv:2211.12588](https://arxiv.org/abs/2211.12588)) — so every *computed* answer routes through one of four tested, pure functions (`get_player_stats`, `rank_players_by_stat`, `compute_team_possession_pct`, `compare_players`) that the model calls as tools. The model decides *which* tool to call and phrases the answer; it never does the arithmetic itself.

**Verification**: `game_report.py`, `llm_client.py`, and `game_qa.py`'s four tool functions have zero ML/GPU dependency and are covered by a real, executed test suite (54 tests, `python3 -m unittest discover -s tests -t .`, all passing — see below). The LLM-call paths themselves are tested against a fake client with scripted responses (see `tests/fakes.py`); I don't currently have a funded API key, so end-to-end output quality against a *real* model hasn't been verified by me yet — that's an honest gap, not a claim.

## 🚀 Quickstart: using the AI insight layer

The video pipeline itself (models, dependencies) is unchanged — see [Installation](#-installation) and [Training the Models](#-training-the-models) further down. This section covers just what's new: turning a processed video into a report, then a narrative or Q&A session.

**1. Get a free API key.** [OpenRouter](https://openrouter.ai/keys) — sign up, no card required, create a key. (Any other provider in `llm_client.py`'s `PROVIDER_CONFIG` — OpenAI, Gemini, DeepSeek — works too, just costs a few cents.)

**2. Set environment variables:**
```bash
export LLM_PROVIDER=openrouter
export LLM_API_KEY=sk-or-...   # your OpenRouter key
```

**3. Run the pipeline with `--report`** to get the video output plus a JSON analytics report (needs the trained models from [Installation](#-installation) in place first):
```bash
python main.py path_to_input_video.mp4 --output_video output_videos/output_result.avi --report output_videos/game_report.json
```

**4. Generate a narrative summary or ask questions about the game:**
```python
import json
from llm_client import get_client
from game_qa import generate_game_narrative, answer_question

report = json.load(open("output_videos/game_report.json"))
client, config = get_client()  # reads LLM_PROVIDER / LLM_API_KEY from the env vars above

print(generate_game_narrative(client, config["model"], report))
print(answer_question(client, config["model"], report, "Who covered the most distance?"))
```

No API key yet? Everything in step 4 is covered by a real, passing test suite run against a fake client (see [Tests](#-tests) below) — you can read it to see exactly what each function does at zero cost.

### Roadmap (researched, not yet built — needs GPU access to build or verify)
Two further upgrades came out of researching current sports-CV practice, both deferred because — unlike the AI layer above — they need real GPU/model access to build or verify at all:
- **Tracking**: swap ByteTrack for [BoT-SORT](https://github.com/mikel-brostrom/boxmot) (motion + camera-compensation), fixing occlusion/re-entry issues plain ByteTrack can't handle.
- **Jersey-number recognition**: OCR on player crops so the report attributes stats to a player number instead of just a team color — the single most differentiating feature per research into current sports-analytics repos (e.g. SoccerNet's official baseline).

## 🧪 Tests

Runs with **zero pip installs** — everything under test is either pure Python or uses dependency-injected fakes (`tests/fakes.py` duck-types the `openai` SDK client shape without needing the package installed):
```
python3 -m unittest discover -s tests -t .
```
53 tests covering: bbox geometry, speed/distance calculation (including a regression test that the real source fps is used, not a hardcoded assumption), a fixed possession-streak-counting bug (see below), pass/interception detection, the `game_report.py` aggregator, and `llm_client.py`/`game_qa.py`'s provider-config and tool-calling logic.

## 🔧 Other fixes in this round
- **Possession-streak counting bug**: `BallAquisitionDetector` used to replace its entire streak-tracking dict every frame, so a single noisy frame for a different candidate reset an otherwise-solid possession streak back to 1. Now streaks decay by one on a miss instead of resetting to zero.
- **Fps correctness**: `save_video` hardcoded a 24fps output regardless of the source video's real frame rate, and `calculate_speed` defaulted to an assumed 30fps — both now read and use the actual source fps.
- **Per-substage profiling**: `--profile`'s single "analysis" bucket used to lump together 6 different pipeline stages (team assignment, ball acquisition, pass/interception, tactical view, speed/distance); it's now broken down per stage so it actually identifies the bottleneck.
- **`requirements.txt`**: removed a redundant duplicate OpenCV pin (`opencv_python` alongside `opencv_python_headless` — only one is needed; no `cv2.imshow`/GUI calls exist anywhere in this codebase).

---

Analyze basketball footage with automated detection of players, ball, team assignment, and more. This repository integrates object tracking, zero-shot classification, and custom keypoint detection for a fully annotated basketball game experience.

Leveraging the convenience of Roboflow for dataset management and Ultralytics' YOLO models for both training and inference, this project provides a robust framework for basketball video analysis.

Training notebooks are included to help you customize and fine-tune models to suit your specific needs, ensuring a seamless and efficient workflow.

## 📁 Table of Contents

1.  [Features](#-features)
2.  [Prerequisites](#-prerequisites)
3.  [Demo Video](#-demo-video)
4.  [Installation](#-installation)
5.  [Training the Models](#-training-the-models)
6.  [Usage](#-usage)
7.  [Project Structure](#-project-structure)
8.  [Future Work](#-future-work)
9.  [Contributing](#-contributing)
10. [License](#-license)

---

## ✨ Features

- Player and ball detection/tracking using pretrained models.
- Court keypoint detection for visualizing important zones.
- Team assignment with jersey color classification.
- Ball possession detection, pass detection, and interception detection.
- Easy stubbing to skip repeated computation for fast iteration.
- Various “drawers” to overlay detected elements onto frames.

---

## 🎮 Demo Video

Below is the final annotated output video.

[![BasketBall Analysis Demo Video](https://img.youtube.com/vi/xWpP0LjEUng/0.jpg)](https://youtu.be/xWpP0LjEUng)

## 🔧 Prerequisites

- Python 3.8+
- (Optional) Docker

---

## ⚙️ Installation

Setup your environment locally or via Docker.

### Python Environment

1. Create a virtual environment (e.g., venv/conda).
2. Install the required packages:

```bash
pip install -r requirements.txt
```

### Docker

#### Build the Docker image:

```bash
docker build -t basketball-analysis .
```

#### Verify the image:

```bash
docker images
```

## 🎓 Training the Models

Harnessing the powerful tools offered by Roboflow and Ultralytics makes it straightforward to manage datasets, handle annotations, and train advanced object detection models. Roboflow provides an intuitive platform for dataset preprocessing and augmentation, while Ultralytics’ YOLO architectures (v5, v8, and beyond) deliver state-of-the-art detection performance.

This repository relies on trained models for detecting basketballs, players, and court keypoints. You have two options to get these models:

1. Download the Pretrained Weights

   - ball_detector_model.pt  
     (https://drive.google.com/file/d/1KejdrcEnto2AKjdgdo1U1syr5gODp6EL/view?usp=sharing)
   - court_keypoint_detector.pt  
     (https://drive.google.com/file/d/1nGoG-pUkSg4bWAUIeQ8aN6n7O1fOkXU0/view?usp=sharing)
   - player_detector.pt  
     (https://drive.google.com/file/d/1fVBLZtPy9Yu6Tf186oS4siotkioHBLHy/view?usp=sharing)

   Simply download these files and place them into the `models/` folder in your project. This allows you to run the pipelines without manually retraining.

2. Train Your Own Models  
   The training scripts are provided in the `training_notebooks/` folder. These Jupyter notebooks use Roboflow datasets and the Ultralytics YOLO frameworks to train various detection tasks:

   - `basketball_ball_training.ipynb`: Trains a basketball ball detector (using YOLOv5). Incorporates motion blur augmentations to improve ball detection accuracy on fast-moving game footage.
   - `basketball_court_keypoint_training.ipynb`: Uses YOLOv8 to detect keypoints on the court (e.g., lines, corners, key zones).
   - `basketball_player_detection_training.ipynb`: Trains a player detection model (using YOLO v11) to identify players in each frame.

   You can easily run these notebooks in Google Colab or another environment with GPU access. After training, download the newly generated `.pt` files and place them in the `models/` folder.

## Once you have your models in place, you may proceed with the usage steps described above. If you want to retrain or fine-tune for your specific dataset, remember to adjust the paths in the notebooks and in `main.py` to point to the newly generated models.

## 🚀 Usage

You can run this repository’s core functionality (analysis pipeline) with Python or Docker.

### 1) Using Python Directly

Run the main entry point with your chosen video file:

```bash
python main.py path_to_input_video.mp4 --output_video output_videos/output_result.avi
```

- By default, intermediate “stubs” (pickled detection results) are used if found, allowing you to skip repeated detection/tracking.
- Use the `--stub_path` flag to specify a custom stub folder, or disable stubs if you want to run everything fresh.

### 2) Using Docker

#### Build the container if not built already:

```bash
docker build -t basketball-analysis .
```

#### Run the container, mounting your local input video folder:

```bash
docker run \
  -v $(pwd)/videos:/app/videos \
  -v $(pwd)/output_videos:/app/output_videos \
  basketball-analysis \
  python main.py videos/input_video.mp4 --output_video output_videos/output_result.avi
```

---

## 🏰 Project Structure

- `main.py`  
  – Orchestrates the entire pipeline: reading video frames, running detection/tracking, team assignment, drawing results, and saving the output video.

- `trackers/`  
  – Houses `PlayerTracker` and `BallTracker`, which use detection models to generate bounding boxes and track objects across frames.

- `utils/`  
  – Contains helper functions like `bbox_utils.py` for geometric calculations, `stubs_utils.py` for reading and saving intermediate results, and `video_utils.py` for reading/saving videos.

- `drawers/`  
  – Contains classes that overlay bounding boxes, court lines, passes, etc., onto frames.

- `ball_aquisition/`  
  – Logic for identifying which player is in possession of the ball.

- `pass_and_interception_detector/`  
  – Identifies passing events and interceptions.

- `court_keypoint_detector/`  
  – Detects lines and keypoints on the court using the specified model.

- `team_assigner/`  
  – Uses zero-shot classification (Hugging Face or similar) to assign players to teams based on jersey color.

- `configs/`  
  – Holds default paths for models, stubs, and output video.

---

## 🔮 Future Work

As we continue to enhance the capabilities of this basketball video analysis tool, several areas for future development have been identified:

1. **Integrating a Pose Model for Advanced Rule Detection**  
   Incorporating a pose detection model could enable the identification of complex basketball rules such as double dribbling and traveling. By analyzing player movements and positions, the system could automatically flag these infractions, adding another layer of analysis to the video footage.

These enhancements will further refine the analysis capabilities and provide users with more comprehensive insights into basketball games.

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a clear explanation of your changes.

---

## 🐜 License

This project is licensed under the MIT License.  
See `LICENSE` for details.

---

## 💬 Questions or Feedback?

Feel free to open an issue or reach out via email if you have questions about the project, suggestions for improvements, or just want to say hi!

Enjoy analyzing basketball footage with automatic detection and tracking!
