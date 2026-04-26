# start with uv/python base image
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# set app dir
WORKDIR /llm-pdf

# copy code
COPY . /llm-pdf
# install requirements
RUN uv sync && uv pip install -e .