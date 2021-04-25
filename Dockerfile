FROM continuumio/miniconda3
WORKDIR /home/biolib


RUN conda install  --yes numpy pandas tensorflow joblib \
    && \
    conda clean -afy


COPY . .

ENTRYPOINT [ "python", "src/predict.py" ]

