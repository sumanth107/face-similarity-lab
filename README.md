# Face Similarity Lab

A transparent Streamlit application for comparing the visual resemblance of two human or
celebrity faces. It detects and aligns the main face in each upload, creates ArcFace embeddings,
computes cosine similarity, and maps the result to a documented 0–100 resemblance score.


> **Not identity verification:** This project reports a heuristic visual-resemblance score. It is
> not suitable for access control, law enforcement, employment, or any other high-stakes decision.

## Example
<img width="731" height="975" alt="image" src="https://github.com/user-attachments/assets/89800802-3b5e-49e4-b1f3-0a3cd2c3fabf" />
<img width="714" height="1005" alt="image" src="https://github.com/user-attachments/assets/fcabc457-3ce2-4ed5-adf4-3c0d32a5fb0f" />



## Features

- Side-by-side JPG, JPEG, MPO, PNG, WEBP, HEIC, and HEIF uploads.
- SCRFD face detection with up to ten detected boxes shown.
- Deterministic main-face selection: largest area, then confidence, then center proximity.
- Five-point alignment and ArcFace ResNet50 embeddings from Buffalo_L.
- Raw cosine similarity plus a documented 0–100 calibration.
- Supporting landmark-geometry comparisons that never alter the main score.
- Detection confidence, quality warnings, aligned crops, model settings, and diagnostic JSON.
- Deterministic human-readable explanations derived from the displayed measurements.
- A deliberately playful score-band roast.

## Architecture

```text
JPG / MPO / PNG / WEBP / HEIC / HEIF
       │
       ▼
Validation + EXIF correction + RGB conversion + bounded resize
       │
       ▼
SCRFD-10GF face detection ──► bounding boxes + confidence + 5 landmarks
       │
       ▼
Main-face selection ──► largest area, then confidence, then center proximity
       │
       ▼
5-point alignment ──► normalized 112 × 112 face crop
       │
       ▼
ArcFace ResNet50 ──► normalized 512-dimensional embedding
       │
       ▼
Cosine similarity ──► logistic calibration ──► optional +9 band adjustment ──► 0–100 score
       │
       ├──► deterministic explanation and quality warnings
       └──► separate landmark-geometry diagnostics (not part of the score)
```

### Models and runtime

The application uses the pretrained **InsightFace Buffalo_L** model pack. Only two modules from
that pack are loaded:

| Pipeline stage | Model | Model file | Output and purpose |
| --- | --- | --- | --- |
| Face detection | SCRFD-10GF | `det_10g.onnx` | Finds faces and returns bounding boxes, detection confidence, and five facial landmarks. |
| Face embedding | ArcFace-style ResNet50 trained on WebFace600K | `w600k_r50.onnx` | Converts one aligned face into a 512-dimensional feature vector. |

Buffalo_L also downloads dense-landmark, 3D-landmark, and age/gender models, but this application
does not load them. `allowed_modules=["detection", "recognition"]` keeps inference focused and
reduces memory use. Both active models run through **ONNX Runtime's CPU execution provider**. No
image is sent to a separate inference API.

The model is initialized lazily and retained with `st.cache_resource`, so Streamlit reruns and
multiple comparisons reuse the same ONNX sessions. The first comparison can take longer because it
may include the automatic Buffalo_L download and ONNX session initialization.

### End-to-end data flow

1. **Decode and validate:** Pillow and `pillow-heif` verify the actual file signature, accept
   JPG/JPEG, MPO, PNG, WEBP, HEIC, and HEIF, apply EXIF orientation, convert to RGB, and reject empty,
   malformed, oversized, or over-20-megapixel uploads.
2. **Bound inference memory:** Images larger than 1600 pixels on their longest side are resized
   while preserving aspect ratio. The original and inference dimensions remain visible in the
   diagnostics.
3. **Detect faces:** SCRFD runs with a `640 × 640` detector input and `0.50` confidence threshold.
   Up to ten faces are retained and displayed.
4. **Select the primary face:** The valid bounding box with the largest area is selected. Equal
   areas are resolved by confidence and then proximity to the image center, making group-photo
   behavior deterministic.
5. **Align the face:** The detected eye, nose, and mouth-corner landmarks define a similarity
   transform. InsightFace produces a standardized `112 × 112` crop so roll, location, and scale
   differences have less effect on the embedding.
6. **Create the embedding:** ArcFace ResNet50 maps the aligned crop to 512 floating-point features.
   The vector is L2-normalized before comparison.
7. **Compare embeddings:** Cosine similarity is the dot product of the two normalized vectors. A
   larger value means the faces are closer in the model's learned feature space.
8. **Calibrate the score:** A documented logistic function maps the raw cosine value to a base
   score. Base scores from 40 through 50 receive a fixed +9 adjustment. Calibration changes
   presentation, not the embedding or raw cosine value.
9. **Explain the result:** Fixed rules report the selected faces, detector confidence, raw cosine,
   calibration substitution, final score band, and input-quality warnings.

### What affects the main score

Only cosine similarity between the two normalized ArcFace embeddings affects the main score.
Detection confidence and quality warnings communicate reliability but never add or subtract
points. The five-landmark geometry table is also separate: it compares visible proportions such
as mouth width and eye-to-nose distance, but it neither changes the score nor claims to expose
ArcFace's internal reasoning.

The full image is used to locate faces. The embedding model receives only the aligned crop, which
substantially reduces sensitivity to outfits, body shape, and background. Results can still be
affected by pose, expression, hairstyle near the crop, makeup, occlusion, aging, lighting, blur,
and demographic bias in the training data.

### Code responsibilities

- `app.py`: Streamlit layout, upload/example selection, cached model lifecycle, session state,
  results, and transparency views.
- `similarity.py`: model initialization, detection, selection, alignment, embeddings, cosine
  similarity, calibration, geometry diagnostics, explanations, and score-band messages.
- `utils.py`: secure image decoding, resizing, RGB/BGR conversion, and bounding-box overlays.

## Requirements

- Python 3.12–3.14 (local tests use 3.12; deployment dependencies include Python 3.14 wheels)
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
similarity and first applies this fixed logistic calibration:

```text
base_score = round(100 / (1 + exp(-8 × (cosine_similarity - 0.10))))
score = base_score
```

The logistic mapping is intentionally resemblance-friendly: a cosine similarity of `0.10` maps
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

## Included example images

The [`test_images`](test_images/) directory contains four attributed Creative Commons portraits.
Choose **Try Examples** inside the app, or select the files manually from the upload
widgets:

- `victoria_justice_2018.png` and `nina_dobrev_2018.png`: similar-looking different people;
  observed score `51` (High).
- `victoria_justice_2018.png` and `victoria_justice_2012.jpg`: same person across images;
  observed score `98` (Very high).
- `nina_dobrev_2018.png` and `nina_dobrev_2011.jpg`: same person across images;
  observed score `96` (Very high).

See [`test_images/README.md`](test_images/README.md) for exact sources, authors, licenses, and
caveats. The values were measured with the pinned model and current calibration; they are not
identity probabilities and can change when the implementation changes.

## Free Streamlit Community Cloud deployment

1. Push this directory to a GitHub repository.
2. Sign in at <https://share.streamlit.io> with GitHub.
3. Choose **Create app** and select the repository, `main` branch, and `app.py` entrypoint.
4. Open **Advanced settings** and select Python 3.14. The pinned compiled dependencies include
   Python 3.14 Linux wheels.
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
- **Multiple faces:** The app outlines up to ten detected faces and uses the largest one. Crop the
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
