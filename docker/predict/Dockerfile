FROM pytorch/pytorch
ARG neptuneToken
ENV NEPTUNE_API_TOKEN ${neptuneToken}
RUN ["pip", "install", "sklearn"]
RUN ["pip", "install", "pytorch-ignite"]
RUN ["pip", "install", "neptune-client"]

