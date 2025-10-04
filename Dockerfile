FROM python:3.10.12-bookworm

WORKDIR /root/code

RUN pip3 install dash
RUN pip3 install pandas
RUN pip3 install dash_bootstrap_components
RUN pip3 install dash-bootstrap-components[pandas]
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install matplotlib
RUN pip3 install mlflow

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
