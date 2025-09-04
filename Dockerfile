FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev \
    build-essential python3-dev \
    ca-certificates curl \
    libfreetype6 libpng16-16 \
    fonts-dejavu-core \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

WORKDIR /app

COPY data_collection/ ./data_collection/
COPY model/ ./model/

# Minimal pinned installs to ensure RichDEM and GDAL compatibility
RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir numpy==1.24.4 && \
    pip install --no-cache-dir GDAL==$(gdal-config --version) && \
    pip install --no-cache-dir joblib==1.3.2 streamlit==1.31.0 matplotlib==3.7.3 shapely==2.0.1 scipy==1.10.1 && \
    pip install --no-cache-dir richdem==0.3.4 --no-build-isolation && \
    pip install --no-cache-dir -r data_collection/requirements.txt && \
    pip cache purge || true && \
    apt-get purge -y libgdal-dev build-essential python3-dev && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p temp_dem data

EXPOSE 8501
HEALTHCHECK --interval=60s --timeout=30s --start-period=600s --retries=5 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "data_collection/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"] 