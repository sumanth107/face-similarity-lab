"""Face detection, embedding comparison, calibration, and explanations."""

from __future__ import annotations

import importlib.metadata
import math
import secrets
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from utils import (
    PreparedImage,
    bgr_to_pil,
    draw_face_overlay,
    pil_to_bgr,
    prepare_for_inference,
)


MODEL_NAME = "buffalo_l"
MODEL_DISPLAY_NAME = "InsightFace Buffalo_L (ArcFace ResNet50)"
DETECTION_THRESHOLD = 0.5
DETECTION_SIZE = (640, 640)
CALIBRATION_CENTER = 0.10
CALIBRATION_SLOPE = 8.0


class FaceSimilarityError(RuntimeError):
    """Base class for expected comparison failures."""


class ModelLoadError(FaceSimilarityError):
    """Raised when InsightFace or its pretrained model cannot be initialized."""


class NoFaceDetectedError(FaceSimilarityError):
    """Raised when no usable face is found in an image."""


class FaceProcessingError(FaceSimilarityError):
    """Raised when a detected face cannot be aligned or embedded."""


@dataclass(frozen=True)
class FaceResult:
    """All non-secret results for one selected face plus its embedding."""

    annotated_image: Image.Image
    aligned_face: Image.Image
    original_size: tuple[int, int]
    inference_size: tuple[int, int]
    resized_for_inference: bool
    detection_count: int
    selected_index: int
    confidence: float
    bbox: tuple[float, float, float, float]
    landmarks: tuple[tuple[float, float], ...]
    all_boxes: tuple[tuple[float, float, float, float], ...]
    embedding: np.ndarray
    quality_warnings: tuple[str, ...]


@dataclass(frozen=True)
class GeometryMetric:
    """One interpretable landmark/bounding-box proportion comparison."""

    name: str
    first_value: float
    second_value: float
    difference_percent: float
    category: str


@dataclass(frozen=True)
class ComparisonResult:
    """Complete result used by the Streamlit presentation layer."""

    first: FaceResult
    second: FaceResult
    cosine_similarity: float
    score: int
    label: str
    reliability: str
    geometry: tuple[GeometryMetric, ...]
    geometry_summary: str
    explanation: str
    roast: str


def create_face_analyzer(
    model_name: str = MODEL_NAME,
    *,
    model_root: str | None = None,
) -> Any:
    """Initialize InsightFace for CPU-only detection and recognition.

    InsightFace downloads Buffalo_L into ``~/.insightface`` on first use.
    Restricting allowed modules prevents gender/age and dense-landmark models
    from being loaded into memory; five detector landmarks are sufficient for
    alignment and the supporting geometry diagnostics.
    """

    try:
        from insightface.app import FaceAnalysis

        kwargs: dict[str, Any] = {
            "name": model_name,
            "allowed_modules": ["detection", "recognition"],
            "providers": ["CPUExecutionProvider"],
        }
        if model_root:
            kwargs["root"] = model_root
        analyzer = FaceAnalysis(**kwargs)
        analyzer.prepare(
            ctx_id=-1,
            det_thresh=DETECTION_THRESHOLD,
            det_size=DETECTION_SIZE,
        )
        models = getattr(analyzer, "models", {})
        if "detection" not in models or "recognition" not in models:
            raise RuntimeError(
                "Buffalo_L did not expose detection and recognition models."
            )
        return analyzer
    except Exception as exc:
        raise ModelLoadError(
            "The face model could not be loaded. On first use, confirm that the server "
            "has internet access and enough disk space for the Buffalo_L download, then retry."
        ) from exc


def select_primary_face(faces: Sequence[Any], image_size: tuple[int, int]) -> int:
    """Select the largest valid face, then confidence, then image-center proximity."""

    image_width, image_height = image_size
    center = np.asarray([image_width / 2.0, image_height / 2.0], dtype=np.float64)
    candidates: list[tuple[tuple[float, float, float], int]] = []

    for index, face in enumerate(faces):
        try:
            bbox = np.asarray(getattr(face, "bbox", None), dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            continue
        if bbox.size < 4 or not np.all(np.isfinite(bbox[:4])):
            continue
        width = max(0.0, float(bbox[2] - bbox[0]))
        height = max(0.0, float(bbox[3] - bbox[1]))
        area = width * height
        if area <= 0:
            continue
        face_center = np.asarray(
            [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0],
            dtype=np.float64,
        )
        distance = float(np.linalg.norm(face_center - center))
        confidence = _finite_float(getattr(face, "det_score", 0.0))
        candidates.append(((area, confidence, -distance), index))

    if not candidates:
        raise NoFaceDetectedError(
            "No usable face was detected. Try a clearer, front-facing photo with one visible face."
        )
    return max(candidates, key=lambda item: item[0])[1]


def analyze_face(image: Image.Image, analyzer: Any) -> FaceResult:
    """Detect the main face, align it, and return a normalized embedding."""

    prepared: PreparedImage = prepare_for_inference(image)
    bgr = pil_to_bgr(prepared.image)
    try:
        faces = list(analyzer.get(bgr, max_num=10, det_metric="default"))
    except Exception as exc:
        raise FaceProcessingError("Face detection failed for this image.") from exc

    if not faces:
        raise NoFaceDetectedError(
            "No face was detected. Try a sharper, front-facing image with less obstruction."
        )

    selected_index = select_primary_face(faces, prepared.inference_size)
    primary = faces[selected_index]
    bbox_array = np.asarray(getattr(primary, "bbox", None), dtype=np.float64).reshape(
        -1
    )
    landmarks_array = np.asarray(getattr(primary, "kps", None), dtype=np.float64)
    embedding_array = np.asarray(
        getattr(primary, "embedding", None), dtype=np.float32
    ).reshape(-1)

    if bbox_array.size < 4:
        raise FaceProcessingError(
            "The detected face did not include a valid bounding box."
        )
    if landmarks_array.shape != (5, 2) or not np.all(np.isfinite(landmarks_array)):
        raise FaceProcessingError(
            "The detected face could not be aligned. Try a more front-facing, unobstructed photo."
        )
    if embedding_array.size == 0 or not np.all(np.isfinite(embedding_array)):
        raise FaceProcessingError("The face model did not produce a valid embedding.")
    norm = float(np.linalg.norm(embedding_array))
    if norm <= 1e-12:
        raise FaceProcessingError("The face model produced an empty embedding.")
    embedding_array = embedding_array / norm

    try:
        from insightface.utils import face_align

        aligned_bgr = face_align.norm_crop(
            bgr,
            landmark=landmarks_array.astype(np.float32),
            image_size=112,
        )
        aligned_face = bgr_to_pil(aligned_bgr)
    except Exception as exc:
        raise FaceProcessingError(
            "The selected face could not be aligned and cropped."
        ) from exc

    boxes = tuple(_box_tuple(getattr(face, "bbox", ())) for face in faces)
    confidences = tuple(
        _finite_float(getattr(face, "det_score", 0.0)) for face in faces
    )
    annotated = draw_face_overlay(prepared.image, boxes, confidences, selected_index)
    bbox = _box_tuple(bbox_array)
    landmarks = tuple((float(point[0]), float(point[1])) for point in landmarks_array)
    warnings = _quality_warnings(
        bbox=bbox,
        landmarks=landmarks_array,
        confidence=confidences[selected_index],
        image_size=prepared.inference_size,
        detection_count=len(faces),
        resized=prepared.resized_for_inference,
    )

    return FaceResult(
        annotated_image=annotated,
        aligned_face=aligned_face,
        original_size=prepared.original_size,
        inference_size=prepared.inference_size,
        resized_for_inference=prepared.resized_for_inference,
        detection_count=len(faces),
        selected_index=selected_index,
        confidence=confidences[selected_index],
        bbox=bbox,
        landmarks=landmarks,
        all_boxes=boxes,
        embedding=embedding_array,
        quality_warnings=warnings,
    )


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    """Return cosine similarity after defensive L2 normalization."""

    vector_a = np.asarray(first, dtype=np.float64).reshape(-1)
    vector_b = np.asarray(second, dtype=np.float64).reshape(-1)
    if vector_a.size == 0 or vector_b.size == 0 or vector_a.shape != vector_b.shape:
        raise ValueError("Embeddings must be non-empty and have matching dimensions.")
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        raise ValueError("Cannot compare a zero-length embedding.")
    similarity = float(np.dot(vector_a / norm_a, vector_b / norm_b))
    return float(np.clip(similarity, -1.0, 1.0))


def calibrate_score(similarity: float) -> int:
    """Map cosine similarity to a resemblance-oriented 0-100 heuristic."""

    if not math.isfinite(similarity):
        raise ValueError("Cosine similarity must be finite.")
    clipped = float(np.clip(similarity, -1.0, 1.0))
    value = 100.0 / (
        1.0 + math.exp(-CALIBRATION_SLOPE * (clipped - CALIBRATION_CENTER))
    )
    return int(np.clip(round(value), 0, 100))


def score_label(score: int) -> str:
    """Return the requested human-readable score band."""

    if not 0 <= score <= 100:
        raise ValueError("Score must be between 0 and 100.")
    if score <= 30:
        return "Low similarity"
    if score <= 50:
        return "Moderate similarity"
    if score <= 70:
        return "High similarity"
    return "Very high similarity"


def compare_geometry(
    first: FaceResult, second: FaceResult
) -> tuple[GeometryMetric, ...]:
    """Compare transparent five-landmark proportions without changing the score."""

    first_metrics = geometry_proportions(first.landmarks, first.bbox)
    second_metrics = geometry_proportions(second.landmarks, second.bbox)
    comparisons: list[GeometryMetric] = []
    for name in first_metrics:
        first_value = first_metrics[name]
        second_value = second_metrics[name]
        denominator = max((abs(first_value) + abs(second_value)) / 2.0, 1e-9)
        difference = abs(first_value - second_value) / denominator * 100.0
        if difference <= 10.0:
            category = "Close"
        elif difference <= 20.0:
            category = "Somewhat different"
        else:
            category = "Different"
        comparisons.append(
            GeometryMetric(name, first_value, second_value, difference, category)
        )
    return tuple(comparisons)


def geometry_proportions(
    landmarks: Sequence[Sequence[float]],
    bbox: Sequence[float],
) -> dict[str, float]:
    """Compute interpretable ratios from InsightFace's five detector landmarks."""

    points = np.asarray(landmarks, dtype=np.float64)
    box = np.asarray(bbox, dtype=np.float64).reshape(-1)
    if points.shape != (5, 2) or box.size < 4:
        raise ValueError(
            "Five landmarks and a four-coordinate bounding box are required."
        )
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(box[:4])):
        raise ValueError("Geometry coordinates must be finite.")

    eye_vector = points[1] - points[0]
    eye_distance = float(np.linalg.norm(eye_vector))
    face_width = float(box[2] - box[0])
    face_height = float(box[3] - box[1])
    if eye_distance <= 1e-9 or face_width <= 1e-9 or face_height <= 1e-9:
        raise ValueError("Face geometry is degenerate.")

    # Rotate landmarks around the eye midpoint so vertical ratios are less
    # sensitive to in-plane head roll.
    eye_midpoint = (points[0] + points[1]) / 2.0
    angle = math.atan2(float(eye_vector[1]), float(eye_vector[0]))
    cosine = math.cos(-angle)
    sine = math.sin(-angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float64)
    rotated = (points - eye_midpoint) @ rotation.T
    mouth_midpoint = (rotated[3] + rotated[4]) / 2.0

    return {
        "Eye spacing / face width": eye_distance / face_width,
        "Mouth width / eye spacing": float(np.linalg.norm(rotated[4] - rotated[3]))
        / eye_distance,
        "Eye line to nose / eye spacing": abs(float(rotated[2, 1])) / eye_distance,
        "Nose to mouth / eye spacing": abs(float(mouth_midpoint[1] - rotated[2, 1]))
        / eye_distance,
        "Face width / height": face_width / face_height,
    }


def compare_faces(
    first: FaceResult, second: FaceResult, *, seed: int | None = None
) -> ComparisonResult:
    """Build the complete comparison result."""

    similarity = cosine_similarity(first.embedding, second.embedding)
    score = calibrate_score(similarity)
    label = score_label(score)
    geometry = compare_geometry(first, second)
    geometry_summary = build_geometry_summary(geometry)
    reliability = _reliability(first, second)
    explanation = build_explanation(
        first,
        second,
        similarity=similarity,
        score=score,
        label=label,
        geometry=geometry,
        reliability=reliability,
    )
    return ComparisonResult(
        first=first,
        second=second,
        cosine_similarity=similarity,
        score=score,
        label=label,
        reliability=reliability,
        geometry=geometry,
        geometry_summary=geometry_summary,
        explanation=explanation,
        roast=roast_message(score, seed),
    )


def build_geometry_summary(metrics: Iterable[GeometryMetric]) -> str:
    """Summarize closest and least-close visible landmark proportions."""

    ordered = sorted(metrics, key=lambda metric: metric.difference_percent)
    if not ordered:
        return "No landmark geometry comparison was available."
    closest = ordered[0]
    furthest = ordered[-1]
    return (
        f"The closest measured proportion is {closest.name.lower()} "
        f"({closest.difference_percent:.1f}% difference). The largest measured difference is "
        f"{furthest.name.lower()} ({furthest.difference_percent:.1f}%). These measurements are "
        "supporting diagnostics and are not inputs to the ArcFace score."
    )


def build_explanation(
    first: FaceResult,
    second: FaceResult,
    *,
    similarity: float,
    score: int,
    label: str,
    geometry: Sequence[GeometryMetric],
    reliability: str,
) -> str:
    """Generate a grounded explanation directly from measured values."""

    close_metrics = [
        metric.name.lower() for metric in geometry if metric.category == "Close"
    ]
    if len(close_metrics) >= 2:
        geometry_sentence = (
            f"The supporting landmark check found close {close_metrics[0]} and "
            f"{close_metrics[1]} proportions."
        )
    elif close_metrics:
        geometry_sentence = (
            f"The supporting landmark check found a close {close_metrics[0]} proportion, "
            "while other measured proportions differed more."
        )
    else:
        geometry_sentence = (
            "The supporting landmark proportions were not especially close, so geometry does "
            "not strongly reinforce the embedding result."
        )

    warning_count = len(first.quality_warnings) + len(second.quality_warnings)
    warning_sentence = (
        f"Reliability is {reliability.lower()}; {warning_count} quality warning(s) are listed below."
        if warning_count
        else f"Reliability is {reliability.lower()} and no obvious input-quality warning was detected."
    )
    score_context = (
        "This suggests a noticeable resemblance, but not an overwhelming one. "
        if 40 <= score <= 50
        else ""
    )
    return (
        f"Both primary faces were detected and aligned. Their normalized ArcFace embeddings have "
        f"a cosine similarity of {similarity:.3f}. Applying the documented calibration maps that "
        f"value to {score}/100 ({label.lower()}). {score_context}{geometry_sentence} "
        f"{warning_sentence} Clothing, "
        "hairstyle, and background are mostly excluded because only aligned face crops are embedded. "
        "This is a resemblance heuristic, not proof that the people share an identity."
    )


def roast_message(score: int, seed: int | None = None) -> str:
    """Choose a roast for the score band, with an optional seed for tests."""

    if score <= 30:
        options = (
            "Bold comparison. The faces disagree.",
            "The resemblance called in sick today.",
            "These two barely share a zip code.",
            "The pixels reviewed your claim and rejected it.",
            "Same species, wildly different patch notes.",
            "That similarity is hiding extremely well.",
            "Even the algorithm asked, ‘Are you sure?’",
            "This match needs a miracle, not calibration.",
        )
    elif score <= 39:
        options = (
            "There's a hint of resemblance, if you squint politely.",
            "I see the idea. The evidence is still buffering.",
            "A little similar, but the jury wants better photos.",
            "The resemblance showed up, but only part-time.",
            "Not impossible — just aggressively subtle.",
            "There's a signal here, but it's on one bar.",
            "Maybe cousins in a very large family.",
            "The model says ‘maybe’; the mirror says ‘keep looking.’",
        )
    elif score <= 50:
        options = (
            "There's definitely some resemblance — just not enough for a double take.",
            "You're onto something. Similar, but not look-alike territory.",
            "A decent match: noticeable, not overwhelming.",
            "The faces rhyme; they don't repeat.",
            "More than coincidence, less than twins.",
            "Good catch — several features line up nicely.",
            "The resemblance is real; it's simply playing it cool.",
            "Close enough to start a debate, not end one.",
        )
    elif score <= 70:
        options = (
            "The resemblance is right there — no zoom required.",
            "Now we're talking. That's a solid match.",
            "Same face energy, different deployment.",
            "A casting director would absolutely pause here.",
            "Not twins, but the algorithm raised an eyebrow.",
            "The resemblance did not come here to be subtle.",
            "This is where the group chat starts arguing.",
            "Close enough to confuse someone across the room.",
        )
    else:
        options = (
            "Be honest — you uploaded the same person twice, didn't you?",
            "The algorithm just did a double take.",
            "At this point, the mirror has questions.",
            "This resemblance is showing off.",
            "Same face card, enterprise edition.",
            "The family group chat would need name tags.",
            "Twins energy. Case closed.",
            "The score is high enough to demand a replay.",
        )
    if seed is None:
        return secrets.choice(options)
    return options[seed % len(options)]


def diagnostics_dict(result: ComparisonResult) -> dict[str, Any]:
    """Create downloadable diagnostics without images or biometric embeddings."""

    return {
        "model": {
            "name": MODEL_DISPLAY_NAME,
            "model_pack": MODEL_NAME,
            "insightface_version": _package_version("insightface"),
            "onnxruntime_version": _package_version("onnxruntime"),
            "provider": "CPUExecutionProvider",
            "detection_threshold": DETECTION_THRESHOLD,
            "detection_size": list(DETECTION_SIZE),
            "aligned_crop_size": [112, 112],
        },
        "first_image": _face_diagnostics(result.first),
        "second_image": _face_diagnostics(result.second),
        "comparison": {
            "cosine_similarity": round(result.cosine_similarity, 6),
            "calibration": ("round(100 / (1 + exp(-8 * (cosine_similarity - 0.10))))"),
            "score": result.score,
            "label": result.label,
            "reliability": result.reliability,
            "geometry_affects_score": False,
            "geometry": [
                {
                    "name": metric.name,
                    "first_value": round(metric.first_value, 6),
                    "second_value": round(metric.second_value, 6),
                    "difference_percent": round(metric.difference_percent, 3),
                    "category": metric.category,
                }
                for metric in result.geometry
            ],
        },
        "privacy": "No image bytes or face embeddings are included in this diagnostic export.",
    }


def _face_diagnostics(face: FaceResult) -> dict[str, Any]:
    return {
        "original_size": list(face.original_size),
        "inference_size": list(face.inference_size),
        "resized_for_inference": face.resized_for_inference,
        "detection_count": face.detection_count,
        "selected_face_number": face.selected_index + 1,
        "selection_rule": "largest area, then confidence, then center proximity",
        "detection_confidence": round(face.confidence, 6),
        "bounding_box": [round(value, 3) for value in face.bbox],
        "five_landmarks": [
            [round(point[0], 3), round(point[1], 3)] for point in face.landmarks
        ],
        "quality_warnings": list(face.quality_warnings),
    }


def _quality_warnings(
    *,
    bbox: tuple[float, float, float, float],
    landmarks: np.ndarray,
    confidence: float,
    image_size: tuple[int, int],
    detection_count: int,
    resized: bool,
) -> tuple[str, ...]:
    warnings: list[str] = []
    image_width, image_height = image_size
    face_width = max(0.0, bbox[2] - bbox[0])
    face_height = max(0.0, bbox[3] - bbox[1])
    face_fraction = (face_width * face_height) / max(image_width * image_height, 1)
    eye_vector = landmarks[1] - landmarks[0]
    eye_distance = max(float(np.linalg.norm(eye_vector)), 1e-9)
    roll = abs(math.degrees(math.atan2(float(eye_vector[1]), float(eye_vector[0]))))
    yaw_proxy = abs(float(landmarks[2, 0] - (landmarks[0, 0] + landmarks[1, 0]) / 2.0))
    yaw_proxy /= eye_distance

    if confidence < 0.70:
        warnings.append(
            "Detection confidence is below 70%; use a clearer face image if possible."
        )
    if face_fraction < 0.03:
        warnings.append("The selected face occupies less than 3% of the image.")
    if roll > 15.0:
        warnings.append(f"The face has approximately {roll:.0f}° of in-plane tilt.")
    if yaw_proxy > 0.18:
        warnings.append(
            "The face may be turned away from the camera, which can affect similarity."
        )
    margin_x = image_width * 0.01
    margin_y = image_height * 0.01
    if (
        bbox[0] <= margin_x
        or bbox[1] <= margin_y
        or bbox[2] >= image_width - margin_x
        or bbox[3] >= image_height - margin_y
    ):
        warnings.append(
            "The selected face is close to an image edge and may be cropped."
        )
    if detection_count > 1:
        warnings.append(
            f"{detection_count} faces were detected; the largest face was used automatically."
        )
    if resized:
        warnings.append(
            "The image was resized to 1600 pixels on its longest side for inference."
        )
    return tuple(warnings)


def _reliability(first: FaceResult, second: FaceResult) -> str:
    warnings = len(first.quality_warnings) + len(second.quality_warnings)
    minimum_confidence = min(first.confidence, second.confidence)
    if minimum_confidence >= 0.85 and warnings == 0:
        return "Strong"
    if minimum_confidence >= 0.70 and warnings <= 2:
        return "Good"
    return "Limited"


def _box_tuple(box: Sequence[float] | np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(box, dtype=np.float64).reshape(-1)
    if values.size < 4 or not np.all(np.isfinite(values[:4])):
        return (0.0, 0.0, 0.0, 0.0)
    return tuple(float(value) for value in values[:4])  # type: ignore[return-value]


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"
