FROM python:3.13

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies needed to build TA-Lib and common Python packages
RUN apt-get update \
     && apt-get install -y --no-install-recommends \
         build-essential \
         wget \
         tar \
         ca-certificates \
         curl \
         git \
         pkg-config \
         autoconf \
         automake \
         libtool \
         gfortran \
         libblas-dev \
         liblapack-dev \
         libffi-dev \
         libssl-dev \
         zlib1g-dev \
         libbz2-dev \
         libreadline-dev \
         libsqlite3-dev \
     && rm -rf /var/lib/apt/lists/*

# Build and install TA-Lib C library (required for Python ta-lib package)
RUN wget -qO /tmp/ta-lib-0.4.0-src.tar.gz https://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
    && mkdir -p /tmp/ta-lib-src \
    && tar -xzf /tmp/ta-lib-0.4.0-src.tar.gz -C /tmp/ta-lib-src --strip-components=1 \
    && cd /tmp/ta-lib-src \
    && ./configure --prefix=/usr \
    && make -j"$(nproc)" \
    && make install \
    && rm -rf /tmp/ta-lib-src /tmp/ta-lib-0.4.0-src.tar.gz

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Default command
CMD ["python", "agent.py"]
