FROM python:3.12-slim
ENV PYTHONPATH=/project/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
WORKDIR /project
COPY requirements-lock.txt pyproject.toml ./
COPY src ./src
# Install the pinned CPU build first so pip does not pull NVIDIA CUDA libraries.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.14.0 \
    && python -m pip install --no-cache-dir -r requirements-lock.txt
COPY . .
# Discard any host bytecode; Python will compile the copied source in Linux.
RUN find /project -type d -name __pycache__ -prune -exec rm -rf {} +
CMD ["python", "run_review.py", "--mode", "verify"]
