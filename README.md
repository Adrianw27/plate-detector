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
4. Generate sythetic data
```
python ./py/gen_data.py
```
5. Train the model
```
python ./py/train.py
```
6. Test the model (choose any image in /data/val)
```
python ./py/infer.py --img data/val/000010.png --ckpt ckpts/crnn.pt
```
7. View prediction

## Dataset source:
@article{RamajoBallester2024,
    title = {Dual license plate recognition and visual features encoding for vehicle identification},
    journal = {Robotics and Autonomous Systems},
    volume = {172},
    pages = {104608},
    year = {2024},
    issn = {0921-8890},
    doi = {https://doi.org/10.1016/j.robot.2023.104608},
    url = {https://www.sciencedirect.com/science/article/pii/S0921889023002476},
    author = {Álvaro Ramajo-Ballester and José María {Armingol Moreno} and Arturo {de la Escalera Hueso}},
    keywords = {Deep learning, Public dataset, ALPR, License plate recognition, Vehicle re-identification, Object detection},
}
