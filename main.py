import cv2
import pytesseract
import numpy as np
from PIL import Image

def preprocess_image(image_path):
    # Read image using OpenCV
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Remove noise
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    # Apply thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def extract_text_from_image(image):
    # Run OCR using pytesseract
    text = pytesseract.image_to_string(image)
    return text

def main():
    image_path = 'test_images/your_image.jpg'  # Update with your image path
    processed = preprocess_image(image_path)
    # Save processed image for debugging
    cv2.imwrite('processed_image.png', processed)
    # Extract text
    text = extract_text_from_image(processed)
    print("Extracted Text:")
    print(text)

if __name__ == "__main__":
    main()