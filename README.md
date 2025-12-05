# License Plate Recognition
Trains a Convolutional Recurrant Neural Network model with Connectionist Temporal Classification loss for Optical Character Recognition tasks, converting license plate photos to text. Developed in Python primarily with PyTorch.

## How to run:
1. Create Python environment in root
```
python3 -m venv .venv
source .venv/bin/activate 
```
3. Install deps
```
pip install --upgrade pip
pip install -r requirements.txt
```
4. Prep data (you can use any dataset here)
```
python ./py/prep_data.py --src sample-data --csv data-labels.csv
```
5. Train the model
```
python ./py/train.py
```
6. Test the model (choose any image in sample-data/)
```
python ./py/infer.py --img sample-data/1000.jpg --ckpt ckpts/crnn.pt
```
7. View prediction
