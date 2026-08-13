FROM public.ecr.aws/lambda/python:3.12
# checkov:skip=CKV_DOCKER_2:HEALTHCHECK is never executed for Lambda container images
# checkov:skip=CKV_DOCKER_3:The Lambda runtime controls the execution user; a USER directive can break the runtime interface client

COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

COPY handler.py ${LAMBDA_TASK_ROOT}/

CMD ["handler.handler"]
