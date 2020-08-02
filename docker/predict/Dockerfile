FROM pytorch/pytorch

WORKDIR /workspace/ml

RUN pip install flask

COPY . /workspace/ml

EXPOSE 80

CMD ["python", "webapp.py"]


