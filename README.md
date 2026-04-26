# LLM PDF: Perform OCR on PDFs using LLM vision models
This project converts pdfs into images (one per page) and sends them to an LLM to extract the content.

# Build image
`docker build -t llm_pdf .`

# Instructions:
1. Set up your openai-compatible LLM connection in the `.env` file.
2. Put your pdf into the `input` folder.

3. Run in docker:
```
docker run --rm \
    -v ./output:/llm-pdf/llm_pdf/output \
    -v ./input:/llm-pdf/llm_pdf/input \
    llm_pdf:latest \
    uv run cli ocr --file_name "test.pdf"
```
4. The extracted markdown file will be located in the `output` folder.