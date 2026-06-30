# Face Similarity Lab

A transparent Streamlit application for comparing the visual resemblance of two human or
celebrity faces. It detects and aligns the main face in each upload, creates ArcFace embeddings,
computes cosine similarity, and maps the result to a documented 0–100 resemblance score.

No paid API, API key, hosted LLM, or remote inference service is used. The app uses InsightFace
Buffalo_L and ONNX Runtime on CPU.

> **Not identity verification:** This project reports a heuristic visual-resemblance score. It is
> not suitable for access control, law enforcement, employment, or any other high-stakes decision.

## Features

- Side-by-side JPG, JPEG, PNG, and WEBP uploads.
- SCRFD face detection with all detected boxes shown.
- Deterministic main-face selection: largest area, then confidence, then center proximity.
- Five-point alignment and ArcFace ResNet50 embeddings from Buffalo_L.
- Raw cosine similarity plus a documented 0–100 calibration.
- Supporting landmark-geometry comparisons that never alter the main score.
- Detection confidence, quality warnings, aligned crops, model settings, and diagnostic JSON.
- Deterministic human-readable explanations with no LLM.
- A deliberately playful score-band roast.

## Architecture

```text
Image Upload → Face Detection → Face Alignment/Cropping → Face Embedding Model
→ Cosine Similarity → Calibrated 0–100 Score → Human-readable Explanation
```

The full image is used only to find faces. Recognition uses a 112×112 aligned face crop, which
mostly removes clothing and background from the comparison.

## Requirements

- Python 3.12 (the tested and deployment-targeted version)
- About 1 GB of free disk space for Python packages and the first-run model download
- Internet access on first model use
- CPU inference; no GPU is required

Buffalo_L is approximately 326 MB and is downloaded automatically to
`~/.insightface/models/buffalo_l` on first use.

## Local installation

macOS/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the URL printed by Streamlit, normally <http://localhost:8501>.

## How scoring works

ArcFace produces a normalized embedding for each aligned face. The app computes cosine
similarity and applies this fixed logistic calibration:

```text
score = round(100 / (1 + exp(-8 × (cosine_similarity - 0.20))))
```

The mapping is intentionally resemblance-friendly: a cosine similarity of `0.20` maps to `50`.
It is a product heuristic, not a probability, biometric match threshold, or population-calibrated
identity score.

| Score | Label |
| ---: | --- |
| 0–30 | Low similarity |
| 31–50 | Moderate similarity |
| 51–70 | High similarity |
| 71–100 | Very high similarity |

The landmark table separately compares eye spacing, mouth width, eye-to-nose distance,
nose-to-mouth distance, and bounding-box aspect ratio. These measurements can make the result
easier to inspect, but they do not reveal ArcFace's internal reasoning and do not affect its score.

## Tests

The automated suite does not download model weights:

```bash
python -m unittest discover -s tests -v
```

To exercise real inference, start the app and compare two clear portraits. Also check a non-face
image, a group image, a profile/occluded face, and malformed input.

## Free Streamlit Community Cloud deployment

1. Push this directory to a GitHub repository.
2. Sign in at <https://share.streamlit.io> with GitHub.
3. Choose **Create app** and select the repository, `main` branch, and `app.py` entrypoint.
4. Open **Advanced settings** and select Python 3.12.
5. Deploy. No secrets or API keys are needed.
6. On the first comparison, allow time for Buffalo_L to download and initialize. Later comparisons
   reuse the process-cached model until the app is restarted or sleeps.

`packages.txt` supplies the small Linux runtime libraries needed by the standard OpenCV wheel.
The app loads only InsightFace's detection and recognition modules and uses `st.cache_resource`
to keep one model instance in memory.

Community Cloud copies the repository and processes uploads on its server. This application does
not intentionally persist images or embeddings and excludes both from diagnostic downloads, but
users should still treat face uploads as sensitive biometric data and obtain appropriate consent.

## Troubleshooting

- **Model load/download failure:** Confirm outbound internet access and free disk space, then
  restart. Remove a partial `~/.insightface/models/buffalo_l` directory only if the download was
  interrupted and remains corrupt.
- **No face detected:** Use a larger, sharper, front-facing face with less occlusion and balanced
  lighting.
- **Multiple faces:** The app outlines every detected face and uses the largest one. Crop the
  intended person before upload if that selection is wrong.
- **Slow first result:** The first result includes the model download and ONNX session startup.
- **`libGL.so` error on Linux:** Install the packages listed in `packages.txt`.
- **Resource-limit error when hosted:** Reboot the app, confirm only one model is cached, and use
  smaller uploads. The 1600-pixel inference cap should prevent image-size memory spikes.

## Limitations and responsible use

Face embeddings and landmark estimates can change because of pose, facial expression, occlusion,
makeup, aging, lighting, blur, compression, and image resolution. Landmark ratios are particularly
sensitive to head pose and bounding-box placement. Face-recognition systems can exhibit unequal
error rates across demographic groups due to dataset and model bias.

Do not use this app to establish identity or make consequential decisions. A high score means only
that these particular aligned images are close under this model and calibration.

## Licensing

The application code can be used under the repository's chosen license. InsightFace's source code
is MIT-licensed, but its supplied pretrained model weights—including Buffalo_L—are restricted to
**non-commercial research use**. Review the
[InsightFace license notice](https://github.com/deepinsight/insightface#license) before publishing
or redistributing the app. Replace the model with appropriately licensed weights before any
commercial use.
