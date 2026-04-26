"""
Simple OCR processor that reads a PDF file path, converts each page to an image,
batches them (max 5 per request), sends them to an OpenAI-compatible endpoint,
and concatenates the results.

Usage:
    ./ocr_processor.py /path/to/document.pdf
    ./ocr_processor.py document.pdf

Environment variables (from .env file):
    OPENAI_API_KEY - API key for OpenAI or compatible service
    OPENAI_API_BASE - Base URL for the API (default: https://api.openai.com/v1)
    OPENAI_MODEL - Model to use (default: gpt-4o)
    OPENAI_MAX_TOKENS - Max tokens in response (optional)
    AZURE_MODE - Set to "true" to use Azure OpenAI
    AZURE_API_KEY - Azure API key (if using Azure)
    AZURE_ENDPOINT - Azure endpoint URL (if using Azure)
    AZURE_DEPLOYMENT - Azure deployment name (if using Azure)

Requirements:
    pip install pymupdf requests python-dotenv
"""
import sys
import base64
import json
from pathlib import Path
from typing import Optional, List
import requests
from dotenv import load_dotenv
import os
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# Import PyMuPDF for PDF rendering
try:
    import fitz  # PyMuPDF
except ImportError:
    print("Error: 'pymupdf' is required but not installed.", file=sys.stderr)
    print("Please install it using: pip install pymupdf", file=sys.stderr)
    sys.exit(1)

def load_env():
    """Load environment variables from .env file in current directory."""
    env_file = Path(".env")
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment from: {env_file.absolute()}", file=sys.stderr)
    else:
        print(f"Warning: .env file not found at {env_file.absolute()}, using system environment variables", file=sys.stderr)
        load_dotenv()

def create_session() -> requests.Session:
    """Create a requests session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def convert_pdf_to_base64_images(file_path: str | Path) -> List[str]:
    """
    Open a PDF, render each page to an image (PNG), and return a list of base64 strings.
    Each string corresponds to one page.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)
    if not file_path.is_file():
        print(f"Error: Path is not a file: {file_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Loading PDF: {file_path.absolute()}", file=sys.stderr)
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening PDF: {e}", file=sys.stderr)
        sys.exit(1)
    
    page_count = len(doc)
    base64_images = []
    for i in range(page_count):
        print(f"Rendering page {i+1}/{page_count}...", file=sys.stderr)
        page = doc[i]
        # Render page to an image (zoom factor 2.0 gives high quality, 1.0 is standard)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        # Save to buffer first
        img_data = pix.tobytes("png")
        # Encode to base64
        b64_str = base64.b64encode(img_data).decode("utf-8")
        base64_images.append(b64_str)
    
    doc.close()
    print(f"Successfully converted {len(base64_images)} pages.", file=sys.stderr)
    return base64_images

def process_batch_openai(
    page_base64_list: List[str],
    api_key: str,
    api_base: str,
    model: str,
    max_tokens: Optional[int] = None
) -> str:
    """
    Send a batch of PDF pages (as base64 images) to OpenAI-compatible endpoint.
    Expected list length <= 5.
    """
    session = create_session()
    endpoint = f"{api_base.rstrip('/')}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
    }
    if api_key and api_key.strip():
        headers["Authorization"] = f"Bearer {api_key}"

    # Construct the content array with multiple images followed by a text prompt
    content_items = []
    
    for i, b64_img in enumerate(page_base64_list):
        content_items.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{b64_img}",
            },
        })

    # Prompt tailored for multi-page processing
    prompt = (f"""
        Transcribe the following screenshot(s) of a document. Try to preserve the layout of each page.
        Format the response in Markdown (e.g., using **bold**, _italics_, `code`, - lists, - [ ] checkboxes and # headings).
        IMPORTANT: Do not wrap your entire output in a markdown codeblock. Output the raw text directly.
        "Process these {len(page_base64_list)} pages in order and always put a Markdown line (---) between each page.
        """
    )

    content_items.append({
        "type": "text",
        "text": prompt,
    })

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": content_items}
        ],
    }
    
    if max_tokens:
        payload["max_tokens"] = max_tokens

    try:
        response = session.post(endpoint, json=payload, headers=headers, timeout=300)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error on batch request: {e.response.status_code}", file=sys.stderr)
        print(f"Response body: {e.response.text[:200]}...", file=sys.stderr)
        raise
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        raise

    result = response.json()
    if "choices" not in result or len(result["choices"]) == 0:
        print(f"Error: Unexpected response format: {result}", file=sys.stderr)
        raise ValueError("No choices in API response")
    
    choice = result["choices"][0]
    # Handle different response formats (gpt-4o usually returns 'message', legacy might return 'text')
    if "message" in choice:
        extracted_text = choice["message"].get("content", "")
    elif "text" in choice:
        extracted_text = choice["text"]
    else:
        extracted_text = choice.get("content", "")
    
    return extracted_text

def main(pdf_file_name):
    """Main entry point."""
    load_env()
    pdf_file_path = Path("llm_pdf") / Path("input") / Path(pdf_file_name)
    
    try:
        # Convert PDF to list of base64 images (one per page)
        print("Converting PDF pages to images...", file=sys.stderr)
        base64_images_list = convert_pdf_to_base64_images(pdf_file_path)
        
        all_texts = []
        api_key = os.getenv("OPENAI_API_KEY", "")
        api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        model = os.getenv("OPENAI_MODEL", "gpt-4o")
        max_tokens_str = os.getenv("OPENAI_MAX_TOKENS")
        max_tokens_val = int(max_tokens_str) if max_tokens_str else None
        
        # Batching configuration
        batch_size = 5
        
        total_pages = len(base64_images_list)
        num_batches = (total_pages + batch_size - 1) // batch_size
        
        print(f"Processing {total_pages} pages in batches of max {batch_size}...", file=sys.stderr)

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, total_pages)
            
            current_batch_images = base64_images_list[start_idx:end_idx]
            
            print(f"Processing batch {i+1}/{num_batches} (pages {start_idx+1}-{end_idx})...", file=sys.stderr)
            
            text = process_batch_openai(
                current_batch_images, 
                api_key, 
                api_base, 
                model, 
                max_tokens_val
            )
            all_texts.append(text)

        # Concatenate results with newlines between pages/batches
        final_output = "\n\n---\n\n".join(all_texts)
        
        # Generate output filename: <original_filename_without_extension>.md
        input_path = Path(pdf_file_path)
        output_name = f"{input_path.stem}.md"
        output_file = Path("llm_pdf") / Path("output") / Path(output_name)
        
        output_file.write_text(final_output)
        print(f"✓ Output saved to: {output_file.absolute()}", file=sys.stderr)
        print(final_output)

    except requests.exceptions.RequestException as e:
        print(f"Error: Request failed: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Unexpected API response format: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main("test.pdf")