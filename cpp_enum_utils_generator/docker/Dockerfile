FROM ubuntu:bionic AS builder

RUN apt update && apt upgrade -y \
 && apt install --no-install-recommends -y curl ca-certificates

WORKDIR /tmp/

RUN curl -o castxml-linux.tar.gz https://data.kitware.com/api/v1/file/63bed74d6d3fc641a02d7e99/download \
 && tar -xvzf castxml-linux.tar.gz

FROM ubuntu:bionic

RUN apt update && apt upgrade -y \
 && apt install --no-install-recommends -y python3 g++ \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash dev
USER dev
WORKDIR /home/dev/

COPY --from=builder /tmp/castxml/ /home/dev/castxml/
