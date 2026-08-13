FROM public.ecr.aws/docker/library/python:3.12-slim
# checkov:skip=CKV_DOCKER_2:HEALTHCHECK is never executed for Lambda container images
# checkov:skip=CKV_DOCKER_3:The Lambda runtime controls the execution user; a USER directive can break the runtime interface client

WORKDIR /opt/ml/code

# Build deps for pandas/pyarrow wheels; plotly+jinja2 are pulled in by evidently.
# linux/amd64 because SageMaker Processing instance-type enum has no ARM
# families (same reason as eval/).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY entrypoint.py .

# ContainerEntrypoint set at CreateProcessingJob time by the handler.
# No CMD needed.
