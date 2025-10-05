FROM python:3.12.11-slim

WORKDIR /root/code

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      mlflow-skinny==2.20.2 \
      defusedxml==0.7.1 \
      matplotlib==3.10.5 \
      numpy==2.3.2 \
      pandas==2.3.2 \
      scikit-learn==1.7.1 \
      dash \
      dash-bootstrap-components \
      dash[testing]

# Testing module
RUN pip3 install dash[testing]

# Set PYTHONPATH
ENV PYTHONPATH="/root/code:${PYTHONPATH}"

# Copy source code
COPY ./code /root/code

# Expose Dash port
EXPOSE 8080

# Start Dash app
CMD ["python3", "/root/code/app.py"]
