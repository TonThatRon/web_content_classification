from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import time
import torch

app = Flask(__name__)

# Load model
MODEL_NAME = "RonTon05/PhoBert_content_256"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def crawl_content(domain):
    # Nếu domain không bắt đầu bằng http:// hoặc https:// thì thêm http://
    if not domain.startswith(("http://", "https://")):
        domain = "http://" + domain

    chrome_options = Options()
    chrome_options.add_argument("--headless")  
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--disable-dev-shm-usage")

    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(domain)

        time.sleep(3)  # Chờ trang tải JS

        html = driver.page_source
        driver.quit()

        soup = BeautifulSoup(html, 'html.parser')

        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()

        texts = soup.stripped_strings
        content = '\n'.join(texts)

        return content

    except Exception as e:
        print(f"Error crawling {domain} with Selenium: {e}")
        return ""

def segment_content(text, max_length=256, stride=128):
    tokens = tokenizer.encode(text, truncation=False)
    segments = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        if len(chunk) < 10:
            continue
        decoded = tokenizer.decode(chunk, skip_special_tokens=True)
        segments.append(decoded)
    return segments

def classify_segments(segments):
    flagged_segments = []
    for segment in segments:
        inputs = tokenizer(segment, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        if prediction == 1:
            flagged_segments.append(segment)
    overall_label = 1 if flagged_segments else 0
    return overall_label, flagged_segments


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    flagged_segments = []
    if request.method == "POST":
        domain = request.form.get("domain")
        content = crawl_content(domain)
        if content:
            segments = segment_content(content)
            label, flagged_segments = classify_segments(segments)
            if label == 1:
                result = f"Domain '{domain}' has label: {label} - malicious content"
            else:
                result = f"Domain '{domain} has label:{label} - normal content"
        else:
            result = f"Failed to crawl content from '{domain}'"
    return render_template("index.html", result=result, flagged_segments=flagged_segments)


if __name__ == "__main__":
    app.run(debug=True)
