FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev python3-gdal proj-bin libproj-dev \
    build-essential python3-dev \
    ca-certificates curl \
    libfreetype6 libpng16-16 \
    fonts-dejavu-core \
    libgeos-dev \
    && rm -rf /var/lib/apt/lists/*

ENV GDAL_DATA=/usr/share/gdal
ENV PROJ_LIB=/usr/share/proj

WORKDIR /app

COPY requirements.txt ./requirements.txt
COPY data_collection/ ./data_collection/
COPY models/ ./models/

# Minimal pinned installs to ensure RichDEM and GDAL compatibility
# Install Python packages in stages to ensure richdem builds properly
RUN pip install --upgrade pip --no-cache-dir

# Install core dependencies first
RUN pip install --no-cache-dir numpy>=1.26.0 scipy>=1.11.0

# Install rasterio first (wheels bundle compatible GDAL); skip pip GDAL to avoid conflicts
RUN pip install --no-cache-dir rasterio==1.3.9

# Install other dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -r data_collection/requirements.txt && \
    pip cache purge || true

# Clean up build dependencies after all packages are installed  
RUN apt-get purge -y libgdal-dev build-essential python3-dev && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir -p temp_dem data

EXPOSE 8501
HEALTHCHECK --interval=60s --timeout=30s --start-period=600s --retries=5 \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "data_collection/app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"] 
