# ocr_service.py

from PIL import Image, ImageEnhance, ImageFilter
import easyocr
import io

def perform_ocr(file_bytes, lang):
    try:
        # Initialize the EasyOCR reader with the specified language(s)
        reader = easyocr.Reader([lang], gpu=True)  # Use GPU if available

        # Read the image from the uploaded file
        img = Image.open(io.BytesIO(file_bytes))

        # Preprocess the image (convert to grayscale, enhance contrast, etc.)
        img = img.convert('L')  # Convert to grayscale
        img = img.filter(ImageFilter.MedianFilter())  # Reduce noise
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2)  # Increase contrast
        img = img.resize((img.width * 2, img.height * 2))  # Upscale the image

        # Convert the PIL image back to bytes for OCR processing
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        # Perform OCR using EasyOCR with the specified language
        result = reader.readtext(img_byte_arr)

        # Extract the text from the OCR result
        extracted_text = ' '.join([item[1] for item in result])
        return extracted_text
    except Exception as e:
        raise RuntimeError(f"Failed to perform OCR: {str(e)}")
