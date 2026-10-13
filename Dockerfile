# MC-3 submission image. Built on the mandated ROCm base image (layer identity
# is checked by the harness before the container runs).
#
# Weights are baked in from ./models (downloaded on the host beforehand):
#   models/llm   -> /models/llm    (vision-language model)
#   models/embed -> /models/embed  (retrieval embedding model)
#
# Build (submission):
#   docker build -t mc3-rag:submit .
#
# Build (local logic test, tiny CPU-friendly model in models/llm-test):
#   docker build -t mc3-rag:test --build-arg LLM_DIR=models/llm-test .
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

WORKDIR /app

ARG LLM_DIR=models/llm
ARG EMBED_DIR=models/embed
ENV MC3_LLM_PATH=/models/llm \
    MC3_EMBED_PATH=/models/embed

# Bake model weights into the image - the container has NO network at eval time.
COPY ${EMBED_DIR} /models/embed
COPY ${LLM_DIR} /models/llm

COPY app/requirements.txt /app/requirements.txt

# Everything torch-adjacent goes in with --no-deps so pip cannot swap the
# base image's ROCm torch for a CUDA build.
RUN pip install --no-cache-dir -r /app/requirements.txt \
 && pip install --no-cache-dir --no-deps \
        "transformers>=4.57" "accelerate>=1.0"

COPY app/ /app/

RUN mkdir -p /app/corpus /app/output /app/index

# Persistent server holding the models + index; app.py talks to it over a unix
# socket. The respawn loop keeps the container (and the service) up for the
# whole evaluation even if the server process ever dies.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD ["sh", "-c", "while true; do python3 /app/server.py; sleep 2; done"]
