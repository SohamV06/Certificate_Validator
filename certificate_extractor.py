from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
import google.generativeai as genai
import fitz  # PyMuPDF
import io
import cv2
import numpy as np
from pyzbar.pyzbar import decode
import requests
from bs4 import BeautifulSoup
import re
from rapidfuzz import fuzz
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import tempfile
import time
from PIL import Image as PILImage
import io

load_dotenv() #load environment variables from .env
genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash')

def pdf_page_to_image(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)  # first page
    pix = page.get_pixmap()
    img_data = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_data))
    return image

def is_certificate(image):
    prompt = "Is this a certificate? Answer yes or no."
    response = model.generate_content([prompt, image])
    parts = response.candidates[0].content.parts
    text = ' '.join(part.text for part in parts).strip().lower()
    if "yes" in text:
        return True
    else:
        return False

def extract_certificate_info(image):
    prompt = "Extract all relevant information from this certificate."
    response = model.generate_content([prompt, image])
    parts = response.candidates[0].content.parts
    text = ' '.join(part.text for part in parts)
    return text

def capture_screenshot_of_url(url):
    options = Options()
    options.headless = True
    options.add_argument("--window-size=1200,800")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    driver = webdriver.Chrome(options=options)
    try:
        driver.get(url)
        time.sleep(5)  # wait for page to load fully, adjust as needed
        screenshot = driver.get_screenshot_as_png()
        return screenshot
    finally:
        driver.quit()

def validate_certificate_url(url, extracted_info):
    try:
        # Capture screenshot of the validation page
        screenshot_png = capture_screenshot_of_url(url)
        # Convert screenshot PNG bytes to PIL Image
        image = PILImage.open(io.BytesIO(screenshot_png))
        # Extract info from screenshot using existing extract_certificate_info function
        extracted_text_from_screenshot = extract_certificate_info(image)
        # Normalize texts for comparison
        norm_extracted_info = re.sub(r'\s+', ' ', extracted_info).lower()
        norm_screenshot_text = re.sub(r'\s+', ' ', extracted_text_from_screenshot).lower()
        # Use fuzzy string matching to compare texts
        similarity = fuzz.ratio(norm_extracted_info, norm_screenshot_text)
        threshold = 70  # similarity threshold percentage
        st.write(f"Validation similarity score from screenshot for URL {url}: {similarity}%")
        if similarity >= threshold:
            return True
        else:
            return False
    except Exception as e:
        st.warning(f"Could not validate certificate URL {url}: {e}")
        return False

def decode_qr_code(image):
    # Convert PIL image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Resize image to 2x for better small QR code detection
    scale_factor = 2
    height, width = cv_image.shape[:2]
    resized_image = cv2.resize(cv_image, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_LINEAR)

    qr_data_set = set()

    # Define tile size and overlap
    tile_size = 300
    overlap = 50

    # Slide over the image in tiles with overlap
    for y in range(0, resized_image.shape[0], tile_size - overlap):
        for x in range(0, resized_image.shape[1], tile_size - overlap):
            tile = resized_image[y:y+tile_size, x:x+tile_size]
            decoded_objects = decode(tile)
            for obj in decoded_objects:
                qr_data_set.add(obj.data.decode('utf-8'))

    # Fallback: try decoding the whole resized image as well
    decoded_objects_full = decode(resized_image)
    for obj in decoded_objects_full:
        qr_data_set.add(obj.data.decode('utf-8'))

    return list(qr_data_set)

##Initialise streamlit app
st.set_page_config(page_title = "Certificate Extractor")
st.header("Certificate Extractor")
uploaded_file = st.file_uploader("Choose a certificate image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

image = None
qr_data = []
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        pdf_bytes = uploaded_file.read()
        try:
            image = pdf_page_to_image(pdf_bytes)
            st.image(image, caption="Certificate Image (from PDF).")
        except Exception as e:
            st.error(f"Could not convert PDF to image: {e}")
    elif uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        image = Image.open(uploaded_file)
        st.image(image, caption="Certificate Image.")
    else:
        st.error("Unsupported file type. Please upload an image or PDF.")

submit = st.button("Extract Certificate Info")

#If extract button is clicked
if submit:
    if image is not None:
        qr_data = decode_qr_code(image)
        if qr_data:
            st.subheader("Decoded QR Code Data")
            for idx, data in enumerate(qr_data):
                st.write(f"QR Code {idx+1}: {data}")
        if is_certificate(image):
            response = extract_certificate_info(image)
            st.subheader("Extracted Information")
            st.write(response)

            # Detect URL in extracted info for validation
            # Updated regex to match URLs with or without http/https scheme
            url_pattern = r'((?:https?://)?(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}(?:/[^\s]*)?)'
            urls = re.findall(url_pattern, response)
            valid = False
            if qr_data:
                # If QR code data is present, skip URL validation
                st.info("QR code detected, skipping URL validation.")
                valid = True
            else:
                for url in urls:
                    # Prepend http:// if scheme is missing
                    if not url.startswith('http://') and not url.startswith('https://'):
                        url = 'http://' + url
                    # Validate URL format before proceeding
                    try:
                        from urllib.parse import urlparse
                        result = urlparse(url)
                        if all([result.scheme, result.netloc]):
                            if validate_certificate_url(url, response):
                                valid = True
                                break
                    except Exception:
                        continue
            if valid:
                st.success("Valid Certificate")
            else:
                st.warning("Certificate validation failed or no matching info found on validation site.")
        else:
            st.error("The given file is not a certificate.")
    else:
        st.error("Please upload a valid certificate image or PDF file.")
