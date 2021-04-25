FROM continuumio/miniconda3
WORKDIR /home/biolib


RUN conda install --yes numpy pandas tensorflow


COPY . .

ENTRYPOINT [ "python", "src/predict.py" ]

