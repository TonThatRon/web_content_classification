from flask import Flask, request, render_template, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from newspaper import Article
import torch

app = Flask(__name__)

# Load model
MODEL_NAME = "RonTon05/PhoBert_content_256"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.eval()

def crawl_content(domain):
    url = f"http://{domain}"
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error: {e}")
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
