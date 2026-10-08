FROM python:3.12-slim
ENV PYTHONPATH=/project/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLBACKEND=Agg
WORKDIR /project
COPY requirements-lock.txt pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir -r requirements-lock.txt
COPY . .
CMD ["python", "run_review.py", "--mode", "verify"]
