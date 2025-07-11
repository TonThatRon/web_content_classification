import torch
from transformers import AutoTokenizer, AutoConfig
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from fastmcp import FastMCP
from fastmcp.server.http import create_sse_app
import asyncio
from models.lexical_domain_content import PhobertLexicalContent
from models.lexical import get_vector_lexical
from models.lexical_extractor import get_combined_lexical_vector

# Initialize FastMCP
mcp = FastMCP("WebClassifierTools")

# Load Tokenizers
tokenizer_phobert = AutoTokenizer.from_pretrained("phunganhsang/PhoBert_Lexical_Dataset55K")
tokenizer_content = AutoTokenizer.from_pretrained("RonTon05/PhoBert_content_48K")

# Load Configs
phobert_config = AutoConfig.from_pretrained("phunganhsang/PhoBert_Lexical_Dataset55K")
content_config = AutoConfig.from_pretrained("RonTon05/PhoBert_content_48K")

# Load Model
model = PhobertLexicalContent(phobert_config, content_config)
model.load_state_dict(torch.load("models/weights/pholexicalcontent_state_dict_30_06.pt", map_location="cpu"))
model.eval()

# Utility Functions
def crawl_content(domain: str) -> str:
    """
    Crawl content from a given URL, removing unnecessary tags like script, style, etc.
    """
    if not domain.startswith(("http://", "https://")):
        domain = "http://" + domain

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    driver = None
    try:
        driver = webdriver.Chrome(options=chrome_options)
        driver.set_page_load_timeout(30)
        driver.get(domain)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        for tag in soup(['script', 'style', 'header', 'footer', 'nav', 'aside']):
            tag.decompose()
        texts = soup.stripped_strings
        return '\n'.join(texts)
    except Exception as e:
        print(f"Error crawling {domain}: {e}")
        return ""
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass

def segment_content(text: str, max_length: int = 200, stride: int = 100) -> list:
    """
    Segment text into chunks for classification.
    """
    tokens = tokenizer_phobert.encode(text, truncation=False)
    segments = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + max_length]
        if len(chunk) < 10:
            continue
        segments.append(tokenizer_phobert.decode(chunk, skip_special_tokens=True))
    return segments

def classify_segments(domain: str, segments: list) -> dict:
    """
    Classify segments using lexical and content features.
    """
    domain_base = domain.split("//")[-1].split("/")[0].split(":")[0]
    lexical = get_vector_lexical(domain_base, return_all=True)
    lexical_vector = get_combined_lexical_vector(domain_base)

    flagged_segments = []

    for idx, segment in enumerate(segments):
        inputs_lexical = tokenizer_phobert(segment, return_tensors="pt", padding="max_length", truncation=True, max_length=200)
        inputs_content = tokenizer_content(segment, return_tensors="pt", padding="max_length", truncation=True, max_length=200)

        with torch.no_grad():
            output = model(
                features=lexical_vector,
                input_ids=inputs_lexical["input_ids"],
                attention_mask=inputs_lexical["attention_mask"],
                input_ids_content=inputs_content["input_ids"],
                attention_mask_content=inputs_content["attention_mask"]
            )
            pred = torch.argmax(output.logits, dim=1).item()

        if idx == 0:
            print("=== SAMPLE SEGMENT ===")
            print("Segment text:\n", segment)
            print("Lexical vector:\n", lexical_vector)
            print("Predicted label:", pred)
            probs = torch.nn.functional.softmax(output.logits, dim=1)
            print("Probabilities:", probs.numpy())

        if pred == 1:
            flagged_segments.append(segment)

    return {
        "label": 1 if flagged_segments else 0,
        "flagged_segments": flagged_segments,
        "lexical_info": lexical
    }

# FastMCP Tool
@mcp.tool("classify_website")
async def classify_website(domain: str) -> dict:
    """
    Classify a website based on its content and lexical features.
    Returns classification label, flagged segments, and lexical information.
    """
    content = crawl_content(domain)
    if not content:
        return {
            "result": f"Không thể crawl nội dung từ '{domain}'",
            "label": None,
            "flagged_segments": [],
            "lexical_info": None
        }

    segments = segment_content(content)
    result_dict = classify_segments(domain, segments)
    label = result_dict["label"]
    flagged_segments = result_dict["flagged_segments"]
    lexical_info = result_dict["lexical_info"]

    return {
        "result": f"Domain '{domain}' được phân loại là: {'TÍNH NHIỆM THẤP' if label == 1 else 'BÌNH THƯỜNG'}",
        "label": label,
        "flagged_segments": flagged_segments,
        "lexical_info": lexical_info
    }

# Create SSE app
app = create_sse_app(
    mcp,
    message_path="/message",  # POST JSON endpoint
    sse_path="/sse"          # SSE endpoint for clients
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)