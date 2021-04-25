FROM continuumio/miniconda3
WORKDIR /home/biolib

RUN pip3 install numpy==1.18.5 -y
RUN pip3 install pandas==1.2.4 -y
RUN pip3 install tensorflow==2.3.0 -y



COPY . .

ENTRYPOINT [ "python", "src/predict.py" ]

