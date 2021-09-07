# Flare21IMTAtlantique
submission of IMT Atlantique for FLARE21 challenge

## training 
- run python3 flare21-train.py -o output_folder

## docker creation and inference (on test FLARE21 data)
- copy epoch.pth in the weights folder
- sudo docker build -t imt_atlantique .
- sudo docker save -o imt_atlantique.tar.gz imt-atlantique
- docker image load < imt_atlantique.tar.gz
- docker container run --gpus "device=0" --name imt_atlantique --rm \
-v $PWD/inputs/:/workspace/inputs/ \
-v $PWD/TeamName_outputs/:/workspace/outputs/ \ 
imt_atlantique:latest /bin/bash -c "sh predict.sh"


