#!/bin/bash
docker run \
    -d \
    --init \
    --rm \
    -p 15001:5000 \
    -p 16007:6006 \
    -p 18512-18522:8501-8511 \
    -p 18888:8888 \
    -it \
    --gpus=all \
    --ipc=host \
    --name=kadai2_docker \
    --env-file=.env \
    --volume=$PWD:/workspace \
    kadai2_docker:latest \
    fish
