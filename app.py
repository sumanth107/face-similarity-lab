"""Streamlit interface for transparent human-face resemblance scoring."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import streamlit as st

from similarity import (
    CALIBRATION_CENTER,
    CALIBRATION_SLOPE,
    ComparisonResult,
    FaceProcessingError,
    ModelLoadError,
    NoFaceDetectedError,
    analyze_face,
    compare_faces,
    create_face_analyzer,
    diagnostics_dict,
)
from utils import ImageValidationError, decode_image, pair_seed


st.set_page_config(
    page_title="Face Similarity Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LOGGER = logging.getLogger(__name__)
APP_DIRECTORY = Path(__file__).resolve().parent
EXAMPLE_PAIRS = {
    "Victoria Justice ↔ Nina Dobrev (similar-looking people)": (
        "victoria_justice_2018.png",
        "nina_dobrev_2018.png",
    ),
    "Victoria Justice 2018 ↔ 2012 (same person)": (
        "victoria_justice_2018.png",
        "victoria_justice_2012.jpg",
    ),
    "Nina Dobrev 2018 ↔ 2011 (same person)": (
        "nina_dobrev_2018.png",
        "nina_dobrev_2011.jpg",
    ),
}

st.markdown(
    """
    <style>
        :root {
            --ink: #24163b;
            --muted-ink: #6f6282;
            --lavender: #7c5bb5;
            --lavender-dark: #5f3f92;
            --lavender-soft: #eee7fa;
            --surface: rgba(255, 255, 255, 0.82);
        }
        html, body, .stApp, [data-testid="stAppViewContainer"],
        button, input, textarea, select {
            font-family: "Avenir Next", Avenir, "Segoe UI", Helvetica, Arial, sans-serif;
        }
        .stApp {
            color: var(--ink);
            background:
                radial-gradient(circle at 12% 0%, rgba(216, 196, 246, 0.56), transparent 31rem),
                linear-gradient(180deg, #f6f1fd 0%, #fbf9fe 56%, #f7f3fc 100%);
        }
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        [data-testid="stAppDeployButton"],
        #MainMenu,
        footer {
            display: none !important;
            visibility: hidden !important;
        }
        .block-container {max-width: 1180px; padding-top: 3.25rem;}
        h1, h2, h3, h4 {
            color: var(--ink);
            font-family: "Avenir Next", Avenir, "Segoe UI", Helvetica, Arial, sans-serif;
            letter-spacing: -0.025em;
        }
        h1 {font-size: clamp(2.35rem, 5vw, 4rem) !important; font-weight: 650 !important;}
        .app-kicker {
            color: var(--lavender-dark);
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            margin-bottom: -0.65rem;
            text-transform: uppercase;
        }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--surface);
            border: 1px dashed #aa93cd;
            border-radius: 0.85rem;
        }
        [data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid rgba(124, 91, 181, 0.18);
            border-radius: 0.85rem;
            padding: 1rem 1.15rem;
        }
        [data-testid="stExpander"] {
            background: rgba(255, 255, 255, 0.68);
            border: 1px solid rgba(124, 91, 181, 0.20);
            border-radius: 0.85rem;
        }
        .score-label {color: var(--lavender-dark); font-size: 1.15rem; font-weight: 700; margin-top: -0.5rem;}
        .roast-box {
            border-left: 5px solid var(--lavender);
            background: rgba(238, 231, 250, 0.88);
            border-radius: 0.4rem;
            padding: 0.85rem 1rem;
            margin: 0.75rem 0 1rem 0;
            font-size: 1.08rem;
        }
        .technical-note {color: var(--muted-ink); font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_face_analyzer():
    """Load one process-wide CPU model instance."""

    return create_face_analyzer()


def _upload_signature(first: bytes | None, second: bytes | None) -> str | None:
    if first is None or second is None:
        return None
    return hashlib.sha256(first + b"\0comparison\0" + second).hexdigest()


def _clear_stale_result(current_signature: str | None) -> None:
    stored_signature = st.session_state.get("comparison_signature")
    if stored_signature != current_signature:
        st.session_state.pop("comparison_result", None)
        st.session_state.pop("comparison_signature", None)


def _show_face_diagnostics(title: str, face) -> None:
    st.markdown(f"#### {title}")
    st.write(f"**Faces detected:** {face.detection_count}")
    st.write(f"**Selected face:** #{face.selected_index + 1} (largest valid face)")
    st.write(f"**Detection confidence:** {face.confidence:.1%}")
    st.write(f"**Original size:** {face.original_size[0]} × {face.original_size[1]} px")
    st.write(
        f"**Inference size:** {face.inference_size[0]} × {face.inference_size[1]} px"
    )
    st.write(
        "**Bounding box:** "
        + ", ".join(f"{coordinate:.1f}" for coordinate in face.bbox)
    )
    st.write("**Aligned/cropped:** Yes — five-point alignment to 112 × 112 px")
    if face.quality_warnings:
        for warning in face.quality_warnings:
            st.warning(warning)
    else:
        st.success("No obvious input-quality warning was detected.")


def _show_result(result: ComparisonResult) -> None:
    score_col, context_col = st.columns([1, 2], vertical_alignment="center")
    with score_col:
        st.metric("Face resemblance score", f"{result.score}/100")
        st.progress(result.score)
        st.markdown(
            f'<div class="score-label">{result.label}</div>', unsafe_allow_html=True
        )
    with context_col:
        st.write(f"**Result reliability:** {result.reliability}")
        st.caption(
            "Reliability describes face-detection and input quality. It does not alter the score."
        )

    st.markdown(
        f'<div class="roast-box"><strong>Playful roast:</strong> {result.roast}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("See exactly how this score was calculated", expanded=False):
        detection_tab, scoring_tab, geometry_tab, architecture_tab = st.tabs(
            ["Detection", "Scoring", "Facial geometry", "Architecture & limitations"]
        )

        with detection_tab:
            first_col, second_col, explanation_col = st.columns([1, 1, 1.25])
            with first_col:
                st.markdown("#### Image A detection")
                st.image(
                    result.first.annotated_image,
                    caption="Green: selected face · Amber: other detected faces",
                    use_container_width=True,
                )
                st.image(result.first.aligned_face, caption="Aligned face A", width=160)
            with second_col:
                st.markdown("#### Image B detection")
                st.image(
                    result.second.annotated_image,
                    caption="Green: selected face · Amber: other detected faces",
                    use_container_width=True,
                )
                st.image(
                    result.second.aligned_face, caption="Aligned face B", width=160
                )
            with explanation_col:
                st.markdown("#### Human-readable explanation")
                st.write(result.explanation)
                st.info(
                    "This explanation is generated from measured values and fixed rules. "
                    "No LLM is used."
                )

            st.divider()
            face_a_col, face_b_col = st.columns(2)
            with face_a_col:
                _show_face_diagnostics("Image A details", result.first)
            with face_b_col:
                _show_face_diagnostics("Image B details", result.second)

        with scoring_tab:
            st.markdown("#### Raw embedding comparison")
            st.metric("Cosine similarity", f"{result.cosine_similarity:.4f}")
            st.write(
                "ArcFace produced one normalized embedding for each aligned face. Cosine "
                "similarity measures how close those vectors are in the learned feature space."
            )
            st.markdown("#### Calibration")
            formula = (
                f"round(100 / (1 + exp(-{CALIBRATION_SLOPE:g} × "
                f"({result.cosine_similarity:.4f} - {CALIBRATION_CENTER:.2f})))) "
                f"= {result.score}"
            )
            st.code(formula, language=None)
            st.caption(
                "This logistic mapping is a resemblance-oriented heuristic, not a probability "
                "and not an identity-verification threshold."
            )
            st.table(
                [
                    {"Score": "0–30", "Label": "Low similarity"},
                    {"Score": "31–50", "Label": "Moderate similarity"},
                    {"Score": "51–70", "Label": "High similarity"},
                    {"Score": "71–100", "Label": "Very high similarity"},
                ]
            )

        with geometry_tab:
            st.markdown("#### Supporting landmark comparison")
            st.warning(
                "These approximate proportions support interpretation but do not affect the "
                "ArcFace score and do not reveal which features ArcFace used."
            )
            st.table(
                [
                    {
                        "Measurement": metric.name,
                        "Image A": f"{metric.first_value:.3f}",
                        "Image B": f"{metric.second_value:.3f}",
                        "Difference": f"{metric.difference_percent:.1f}%",
                        "Assessment": metric.category,
                    }
                    for metric in result.geometry
                ]
            )
            st.write(result.geometry_summary)
            st.caption(
                "Close: ≤10% difference · Somewhat different: >10–20% · Different: >20%. "
                "Pose, expression, and bounding-box placement can change these measurements."
            )

        with architecture_tab:
            st.markdown("#### Processing pipeline")
            st.code(
                "Image Upload → Face Detection → Face Alignment/Cropping → Face Embedding "
                "Model → Cosine Similarity → Calibrated 0–100 Score → Human-readable Explanation",
                language=None,
            )
            st.markdown("#### Components")
            st.write(
                "**Detection:** InsightFace SCRFD · **Alignment:** five facial landmarks · "
                "**Embedding:** Buffalo_L ArcFace ResNet50 · **Inference:** ONNX Runtime CPU"
            )
            st.markdown("#### What is and is not compared")
            st.write(
                "The model embeds aligned face crops, so clothing, body shape, and background are "
                "mostly excluded. Hairstyle can still influence the crop edges. The result can be "
                "affected by pose, expression, occlusion, makeup, aging, lighting, resolution, and "
                "demographic biases present in face-recognition training data."
            )
            st.error(
                "This is a visual-similarity tool, not a definitive identity-verification, access-control, "
                "law-enforcement, employment, or other high-stakes decision system."
            )
            st.caption(
                "Buffalo_L model weights are provided by InsightFace for non-commercial research use."
            )

        diagnostic_json = json.dumps(diagnostics_dict(result), indent=2)
        st.download_button(
            "Download diagnostics (JSON)",
            diagnostic_json,
            file_name="face-similarity-diagnostics.json",
            mime="application/json",
            help="Contains measurements and settings, but no images or face embeddings.",
        )


st.markdown('<p class="app-kicker">Portrait comparison</p>', unsafe_allow_html=True)
st.title("Face Similarity")

input_source = st.radio(
    "Choose image source",
    ("Upload images", "Try a bundled example"),
    horizontal=True,
)
upload_col_a, upload_col_b = st.columns(2)
bytes_a: bytes | None = None
bytes_b: bytes | None = None
caption_a = "Original image A"
caption_b = "Original image B"

if input_source == "Upload images":
    with upload_col_a:
        upload_a = st.file_uploader(
            "Upload image A",
            type=["jpg", "jpeg", "png", "webp"],
            key="upload_a",
            help="Maximum 10 MB and 20 megapixels.",
        )
    with upload_col_b:
        upload_b = st.file_uploader(
            "Upload image B",
            type=["jpg", "jpeg", "png", "webp"],
            key="upload_b",
            help="Maximum 10 MB and 20 megapixels.",
        )
    bytes_a = upload_a.getvalue() if upload_a is not None else None
    bytes_b = upload_b.getvalue() if upload_b is not None else None
else:
    example_name = st.selectbox("Example pair", tuple(EXAMPLE_PAIRS))
    filename_a, filename_b = EXAMPLE_PAIRS[example_name]
    try:
        bytes_a = (APP_DIRECTORY / "test_images" / filename_a).read_bytes()
        bytes_b = (APP_DIRECTORY / "test_images" / filename_b).read_bytes()
        caption_a = filename_a.replace("_", " ").rsplit(".", 1)[0].title()
        caption_b = filename_b.replace("_", " ").rsplit(".", 1)[0].title()
    except OSError:
        st.error("The bundled example images are missing from this deployment.")
    st.caption(
        "Bundled portraits are Creative Commons images from Wikimedia Commons. "
        "Attribution and license details are in test_images/README.md."
    )

signature = _upload_signature(bytes_a, bytes_b)
_clear_stale_result(signature)

image_a = None
image_b = None
if bytes_a is not None:
    try:
        image_a = decode_image(bytes_a)
        with upload_col_a:
            st.image(image_a, caption=caption_a, use_container_width=True)
    except ImageValidationError as exc:
        with upload_col_a:
            st.error(str(exc))
if bytes_b is not None:
    try:
        image_b = decode_image(bytes_b)
        with upload_col_b:
            st.image(image_b, caption=caption_b, use_container_width=True)
    except ImageValidationError as exc:
        with upload_col_b:
            st.error(str(exc))

ready = (
    image_a is not None
    and image_b is not None
    and bytes_a is not None
    and bytes_b is not None
)
compare_clicked = st.button(
    "Compare faces",
    type="primary",
    disabled=not ready,
    use_container_width=True,
)

if not ready:
    st.info("Upload two valid face images or choose a bundled example to begin.")

if (
    compare_clicked
    and ready
    and signature is not None
    and bytes_a is not None
    and bytes_b is not None
):
    # Do not leave a previous result visible when retrying the same files.
    st.session_state.pop("comparison_result", None)
    st.session_state.pop("comparison_signature", None)
    try:
        with st.spinner(
            "Loading the face model and comparing images. The first run may download Buffalo_L…"
        ):
            analyzer = get_face_analyzer()
            try:
                first_result = analyze_face(image_a, analyzer)
            except NoFaceDetectedError as exc:
                raise NoFaceDetectedError(f"Image A: {exc}") from exc
            except FaceProcessingError as exc:
                raise FaceProcessingError(f"Image A: {exc}") from exc
            try:
                second_result = analyze_face(image_b, analyzer)
            except NoFaceDetectedError as exc:
                raise NoFaceDetectedError(f"Image B: {exc}") from exc
            except FaceProcessingError as exc:
                raise FaceProcessingError(f"Image B: {exc}") from exc
            comparison = compare_faces(
                first_result,
                second_result,
                seed=pair_seed(bytes_a, bytes_b),
            )
        st.session_state["comparison_result"] = comparison
        st.session_state["comparison_signature"] = signature
    except NoFaceDetectedError as exc:
        st.error(f"Face detection stopped the comparison: {exc}")
    except FaceProcessingError as exc:
        st.error(f"A detected face could not be processed: {exc}")
    except ModelLoadError as exc:
        LOGGER.exception("Face model failed to load")
        st.error(str(exc))
    except Exception:
        LOGGER.exception("Unexpected face-comparison failure")
        st.error(
            "An unexpected comparison error occurred. Try smaller, clearer face images or restart the app."
        )

stored_result = st.session_state.get("comparison_result")
stored_signature = st.session_state.get("comparison_signature")
if isinstance(stored_result, ComparisonResult) and stored_signature == signature:
    st.divider()
    _show_result(stored_result)
