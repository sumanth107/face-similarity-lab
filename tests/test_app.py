from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


class StreamlitSmokeTests(unittest.TestCase):
    def test_app_starts_without_loading_the_model(self) -> None:
        app = AppTest.from_file("app.py")
        app.run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("file_uploader")), 2)
        self.assertEqual(len(app.button), 1)
        self.assertTrue(app.button[0].disabled)

    def test_bundled_example_mode_loads_images_without_model_inference(self) -> None:
        app = AppTest.from_file("app.py")
        app.run(timeout=30)
        app.radio[0].set_value("Try Examples").run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.selectbox), 1)
        self.assertEqual(len(app.get("file_uploader")), 0)
        self.assertFalse(app.button[0].disabled)


if __name__ == "__main__":
    unittest.main()
