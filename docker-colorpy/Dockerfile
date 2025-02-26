### CURVELINK DOCKERFILE ###
### SWAPLINK DOCKERFILE ###

# Verwende ein offizielles Python-Image als Basis
FROM python:3.12-slim

# Aktualisiere und installiere Systemabhängigkeiten
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

# Erstelle und verwende ein virtuelles Environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Installiere Python-Bibliotheken
RUN pip install --upgrade pip
RUN pip install awslambdaric boto3 botocore numpy scipy orjson

# Setze den Arbeitsverzeichnis
WORKDIR /app

# Kopiere den src Ordner ins Arbeitsverzeichnis
ADD src/ ./src/

# Set runtime interface client as default command for the container runtime
ENTRYPOINT [ "python", "-m", "awslambdaric" ]
# Pass the name of the function handler as an argument to the runtime
CMD [ "src/lambda_function.lambda_handler" ]
