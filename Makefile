SHELL := /bin/bash
.DEFAULT_GOAL := help
DOCKER_IMAGE := cnn_image_recog
DOCKER_TAG := latest
VENV_DIR := .venv

help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)";echo;sed -ne"/^## /{h;s/.*//;:d" -e"H;n;s/^## //;td" -e"s/:.*//;G;s/\\n## /---/;s/\\n/ /g;p;}" ${MAKEFILE_LIST}|LC_ALL='C' sort -f|awk -F --- -v n=$$(tput cols) -v i=19 -v a="$$(tput setaf 6)" -v z="$$(tput sgr0)" '{printf"%s%*s%s ",a,-i,$$1,z;m=split($$2,w," ");l=n-i;for(j=1;j<=m;j++){l-=length(w[j])+1;if(l<= 0){l=n-i-length(w[j])-1;printf"\n%*s ",-i," ";}printf"%s ",w[j];}printf"\n";}'|more $(shell test $(shell uname) == Darwin && echo '-Xr')

## build the python virtual env for the project
venv:
	uv venv --python 3.12
	uv pip install pip setuptools wheel poetry
	${VENV_DIR}/bin/poetry export -f requirements.txt --output requirements.txt
	uv pip install -r ./requirements.txt
	rm ./requirements.txt
	uv pip install -e .

## build the model
build:
	${VENV_DIR}/bin/python cnn_image_recog/cnn.py --build

## test the model
test:
	${VENV_DIR}/bin/python cnn_image_recog/cnn.py --test