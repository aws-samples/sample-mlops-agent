FROM public.ecr.aws/docker/library/python:3.12-slim
# checkov:skip=CKV_DOCKER_2:HEALTHCHECK is never executed for Lambda container images
# checkov:skip=CKV_DOCKER_3:The Lambda runtime controls the execution user; a USER directive can break the runtime interface client

WORKDIR /opt/ml/code

# Build deps for pyarrow/pandas wheels. Image is built linux/amd64
# because SageMaker Processing instance-type enum has no ARM families;
# see cdk/lib/constructs/arm-build-construct.ts.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY entrypoint.py schema_transform.py ./

# SageMaker Processing launches the container via the ScriptProcessor
# command + code path; the ContainerEntrypoint the launcher passes runs
# `python entrypoint.py`. No CMD needed.
