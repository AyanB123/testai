import pytesseract
import latexcodec
import cv2
from PIL import Image
import requests

# Replace the URL with the path to your image
url = "https://example.com/image.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content))

# Convert image to grayscale
img = img.convert('L')

# Apply image thresholding
img = img.point(lambda x: 0 if x < 128 else 255, '1')

# Extract text using OCR
text = pytesseract.image_to_string(img)

latex_text = latexcodec.encode(text)
print(latex_text)
latex_text = latexcodec.encode(text)

# Print the extracted text
print(text)
print(latex_text)
