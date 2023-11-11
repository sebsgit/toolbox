FROM ubuntu:bionic

RUN apt update && apt upgrade -y \
 && apt install -y python3 curl g++ \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash dev
USER dev
WORKDIR /home/dev/

RUN curl -o castxml-linux.tar.gz https://data.kitware.com/api/v1/file/63bed74d6d3fc641a02d7e99/download \
 && tar -xvzf castxml-linux.tar.gz \
 && rm castxml-linux.tar.gz 
