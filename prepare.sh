#!/bin/bash

echo "[*] Installing gdown"
pip install gdown

json_dir="./data/VRD/json_dataset"
wordvec_dir="./data/wordvectors"
weights_dir="./weights"

echo "[*] Downloading VRD dataset"
mkdir -p $json_dir
wget https://cs.stanford.edu/people/ranjaykrishna/vrd/json_dataset.zip -P $json_dir
# check if unzip is installed
if ! [ -x "$(command -v unzip)" ]; then
  echo '[-] Error: unzip is not installed.' >&2
  exit 1
fi
unzip $json_dir/json_dataset.zip -d $json_dir

echo "[*] Downloading weights"
mkdir -p $weights_dir
# https://drive.google.com/file/d/1UJXqYKEx9BvsYNvksQQQlaUcTQr_MhDb/view
gdown https://drive.google.com/uc?id=1UJXqYKEx9BvsYNvksQQQlaUcTQr_MhDb -O $weights_dir/S3D_kinetics400.pt

echo "[*] Downloading word2vectors"
# https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/view?usp=drive_link&resourcekey=0-wjGZdNAUop6WykTtMip30g
mkdir -p $wordvec_dir
gdown https://drive.google.com/uc?id=0B7XkCwpI5KDYNlNUTTlSS21pQmM -O $wordvec_dir/GoogleNews-vectors-negative300.bin.gz 
# check if gunzip is installed
if ! [ -x "$(command -v gunzip)" ]; then
  echo '[-] Error: gunzip is not installed.' >&2
  exit 1
fi
gunzip $wordvec_dir/GoogleNews-vectors-negative300.bin.gz


