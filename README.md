# Flare21IMTAtlantique
submission of IMT Atlantique for FLARE21 challenge

author: Pierre-Henri Conze | IMT Atlantique, LaTIM UMR 1101, Inserm

## training 
```python
python3 flare21-train.py -i input_folder -o output_folder
```

for training, note that ```input_folder``` should contain both ```TrainingImg/``` and ```TrainingMask/``` folders

## docker creation 
- copy epoch.pth in the weights folder
```python
sudo docker build -t imt_atlantique .
```
```python
sudo docker save -o imt_atlantique.tar.gz imt_atlantique
```

## inference on test FLARE21 data
```python
sudo docker image load < imt_atlantique.tar.gz
- docker container run --gpus "device=0" --name imt_atlantique --rm \
-v $PWD/inputs/:/workspace/inputs/ \
-v $PWD/TeamName_outputs/:/workspace/outputs/ \ 
imt_atlantique:latest /bin/bash -c "sh predict.sh"
```


