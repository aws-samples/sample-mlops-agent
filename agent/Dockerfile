FROM public.ecr.aws/docker/library/python:3.12-slim
WORKDIR /app

# Node.js is required: claude_agent_sdk runs the Claude CLI as a Node.js subprocess.
# git is required for Claude Code's Bash tool.
# gcc/g++ for native Python extensions.
# libreoffice + poppler for document preview (Word/PPT/Excel → PDF → image).
# libnss3/libnspr4/libdbus-1-3 for headless browser support (Playwright).
RUN apt-get update && apt-get install -y \
    gcc g++ \
    ca-certificates curl git \
    libnss3 libnspr4 libdbus-1-3 \
    libreoffice-writer libreoffice-impress \
    poppler-utils \
    fonts-nanum \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install Claude Code CLI via npm so the SDK uses system Node.js 22 instead of
# the bundled standalone binary (which silently exits with code 1 in this container).
RUN npm install -g @anthropic-ai/claude-code
COPY . .

# claude_agent_sdk passes --dangerously-skip-permissions which is blocked as root.
RUN useradd -m -u 1000 -s /bin/bash agent && chown -R agent:agent /app \
 && if [ -d /app/.claude ]; then cp -r /app/.claude /home/agent/.claude && chown -R agent:agent /home/agent/.claude; fi
USER agent

# Use Bedrock + IAM execution role for Claude CLI auth (no Anthropic API key needed).
ENV CLAUDE_CODE_USE_BEDROCK=1
ENV PORT=8080
# OTEL env vars (distro, configurator, observability toggle) are set by CDK
# so the agent image stays portable across stacks.
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["opentelemetry-instrument", "python", "main.py"]
