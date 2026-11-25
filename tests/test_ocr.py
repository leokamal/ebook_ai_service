import io

import pytest
from PIL import Image

import ocr_service


@pytest.mark.parametrize("size, expected", [((100, 50), (100, 200)), ((1600, 900), (900, 1600))])
def test_only_small_images_are_upscaled(monkeypatch, size, expected):
    seen = []

    class Reader:
        def readtext(self, image):
            seen.append(image)
            return [(None, "hello", 1.0), (None, "world", 1.0)]

    monkeypatch.setattr(ocr_service, "get_reader", lambda lang: Reader())
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")

    assert ocr_service.perform_ocr(buffer.getvalue(), "en") == "hello world"
    assert seen[0].shape == expected  # grayscale array (height, width)
