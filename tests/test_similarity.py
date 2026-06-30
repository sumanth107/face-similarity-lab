from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

from similarity import (
    FaceResult,
    NoFaceDetectedError,
    analyze_face,
    build_geometry_summary,
    calibrate_score,
    compare_faces,
    cosine_similarity,
    diagnostics_dict,
    geometry_proportions,
    roast_message,
    score_label,
    select_primary_face,
)


def contains_key(value: object, forbidden_key: str) -> bool:
    if isinstance(value, dict):
        return forbidden_key in value or any(
            contains_key(nested, forbidden_key) for nested in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_key(nested, forbidden_key) for nested in value)
    return False


def make_face_result(
    embedding: np.ndarray,
    *,
    confidence: float = 0.95,
    warnings: tuple[str, ...] = (),
) -> FaceResult:
    image = Image.new("RGB", (200, 200), "white")
    landmarks = (
        (70.0, 80.0),
        (130.0, 80.0),
        (100.0, 110.0),
        (80.0, 140.0),
        (120.0, 140.0),
    )
    return FaceResult(
        annotated_image=image,
        aligned_face=image.resize((112, 112)),
        original_size=(200, 200),
        inference_size=(200, 200),
        resized_for_inference=False,
        detection_count=1,
        selected_index=0,
        confidence=confidence,
        bbox=(40.0, 40.0, 160.0, 180.0),
        landmarks=landmarks,
        all_boxes=((40.0, 40.0, 160.0, 180.0),),
        embedding=np.asarray(embedding, dtype=np.float32),
        quality_warnings=warnings,
    )


class SimilarityTests(unittest.TestCase):
    def test_cosine_similarity_identity_and_orthogonal(self) -> None:
        self.assertAlmostEqual(
            cosine_similarity(np.array([2, 0]), np.array([7, 0])), 1.0
        )
        self.assertAlmostEqual(
            cosine_similarity(np.array([1, 0]), np.array([0, 1])), 0.0
        )

    def test_cosine_similarity_rejects_invalid_embeddings(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity(np.array([0, 0]), np.array([1, 0]))
        with self.assertRaises(ValueError):
            cosine_similarity(np.array([1, 2]), np.array([1, 2, 3]))

    def test_calibration_is_bounded_monotonic_and_centered(self) -> None:
        values = [calibrate_score(value) for value in np.linspace(-1.0, 1.0, 101)]
        self.assertEqual(values, sorted(values))
        self.assertGreaterEqual(min(values), 0)
        self.assertLessEqual(max(values), 100)
        self.assertEqual(calibrate_score(0.10), 50)
        self.assertEqual(calibrate_score(0.1071), 51)

    def test_score_label_boundaries(self) -> None:
        expected = {
            0: "Low similarity",
            30: "Low similarity",
            31: "Moderate similarity",
            50: "Moderate similarity",
            51: "High similarity",
            70: "High similarity",
            71: "Very high similarity",
            100: "Very high similarity",
        }
        for score, label in expected.items():
            self.assertEqual(score_label(score), label)

    def test_primary_face_prefers_area_then_confidence_then_center(self) -> None:
        faces = [
            SimpleNamespace(bbox=np.array([0, 0, 40, 40]), det_score=0.99),
            SimpleNamespace(bbox=np.array([20, 20, 100, 100]), det_score=0.70),
        ]
        self.assertEqual(select_primary_face(faces, (120, 120)), 1)

        tied_area = [
            SimpleNamespace(bbox=np.array([0, 0, 50, 50]), det_score=0.80),
            SimpleNamespace(bbox=np.array([25, 25, 75, 75]), det_score=0.90),
        ]
        self.assertEqual(select_primary_face(tied_area, (100, 100)), 1)

        tied_confidence = [
            SimpleNamespace(bbox=np.array([0, 0, 50, 50]), det_score=0.90),
            SimpleNamespace(bbox=np.array([25, 25, 75, 75]), det_score=0.90),
        ]
        self.assertEqual(select_primary_face(tied_confidence, (100, 100)), 1)

    def test_primary_face_ignores_invalid_boxes(self) -> None:
        faces = [
            SimpleNamespace(bbox=None, det_score=0.99),
            SimpleNamespace(bbox=np.array([10, 10, 40, 40]), det_score=0.80),
        ]
        self.assertEqual(select_primary_face(faces, (100, 100)), 1)
        with self.assertRaises(NoFaceDetectedError):
            select_primary_face([SimpleNamespace(bbox=None)], (100, 100))

    def test_analyze_face_reports_no_face_cleanly(self) -> None:
        analyzer = SimpleNamespace(get=lambda *_args, **_kwargs: [])
        with self.assertRaisesRegex(NoFaceDetectedError, "No face was detected"):
            analyze_face(Image.new("RGB", (100, 100), "white"), analyzer)

    def test_geometry_is_rotation_tolerant(self) -> None:
        landmarks = np.array(
            [[70, 80], [130, 80], [100, 110], [80, 140], [120, 140]], dtype=float
        )
        metrics = geometry_proportions(landmarks, (40, 40, 160, 180))
        self.assertAlmostEqual(metrics["Eye spacing / face width"], 0.5)
        self.assertAlmostEqual(metrics["Mouth width / eye spacing"], 2 / 3)
        self.assertAlmostEqual(metrics["Eye line to nose / eye spacing"], 0.5)
        self.assertAlmostEqual(metrics["Nose to mouth / eye spacing"], 0.5)

    def test_complete_comparison_and_diagnostics_exclude_embeddings(self) -> None:
        first = make_face_result(np.array([1.0, 0.0, 0.0]))
        second = make_face_result(np.array([0.8, 0.2, 0.0]))
        result = compare_faces(first, second, seed=7)
        self.assertGreater(result.score, 70)
        self.assertIn("cosine similarity", result.explanation)
        self.assertIn("not inputs", build_geometry_summary(result.geometry))
        diagnostics = diagnostics_dict(result)
        self.assertFalse(contains_key(diagnostics, "embedding"))
        self.assertFalse(contains_key(diagnostics, "image_bytes"))

    def test_upper_moderate_score_has_encouraging_context(self) -> None:
        first = make_face_result(np.array([1.0, 0.0]))
        cosine = 0.075
        second = make_face_result(np.array([cosine, np.sqrt(1.0 - cosine**2)]))
        result = compare_faces(first, second, seed=0)
        self.assertGreaterEqual(result.score, 40)
        self.assertLessEqual(result.score, 50)
        self.assertIn(
            "noticeable resemblance, but not an overwhelming one", result.explanation
        )

    def test_roast_is_stable_and_band_specific(self) -> None:
        self.assertEqual(roast_message(90, 123), roast_message(90, 123))
        self.assertNotEqual(roast_message(10, 0), roast_message(90, 0))
        for score in (10, 35, 45, 60, 90):
            options = {roast_message(score, seed) for seed in range(8)}
            self.assertEqual(len(options), 8)
        self.assertIn("resemblance is definitely there", roast_message(45, 0))


if __name__ == "__main__":
    unittest.main()
