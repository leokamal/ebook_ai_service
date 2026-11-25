# ocr_service.py

from PIL import Image, ImageEnhance, ImageFilter
import io
from functools import lru_cache

# Smaller images are upscaled 2x to help recognition. 2 x 1280 matches EasyOCR's default
# canvas size (2560), so upscaling anything larger mostly costs time.
UPSCALE_BELOW = 1280


@lru_cache(maxsize=4)
def get_reader(lang):
    import easyocr  # lazy: importing torch/easyocr is slow and would delay port binding
    return easyocr.Reader([lang], gpu=False)

def perform_ocr(file_bytes, lang):
    try:
        reader = get_reader(lang)

        # Read the image from the uploaded file
        img = Image.open(io.BytesIO(file_bytes))

        # Preprocess the image (convert to grayscale, enhance contrast, etc.)
        img = img.convert('L')  # Convert to grayscale
        img = img.filter(ImageFilter.MedianFilter())  # Reduce noise
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)  # Increase contrast
        if max(img.size) < UPSCALE_BELOW:
            img = img.resize((img.width * 2, img.height * 2))  # Upscale small images

        # Perform OCR using EasyOCR with the specified language.
        # Pixels are passed directly: no PNG encode here and decode inside EasyOCR.
        import numpy as np
        result = reader.readtext(np.asarray(img))

        # Extract the text from the OCR result
        extracted_text = ' '.join([item[1] for item in result])
        return extracted_text
    except Exception as e:
        raise RuntimeError(f"Failed to perform OCR: {str(e)}")
