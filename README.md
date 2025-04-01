# ZeaMays-Seed-Detection-LowLight
A deep learning-based object detection model for identifying Zea mays seeds in low-light conditions using SSD MobileNet. The model is optimized for accuracy in challenging lighting environments.

- **Pre-trained model :** SSDMobileNetV2-with FPN
- **API :** Tensorflow Object Detection API
- **Dataset**: [GermPredDataset.zip (data.mendeley.com)](https://data.mendeley.com/datasets/4wkt6thgp6/2/files/4ab46f9f-b34f-4112-a4be-0f3d6c459cf4)

Traied checkpoints are given in
```bash
zeamays-model/
```

## Requirements
Install following dependencies:
```bash
pip install tensorflow opencv-python numpy matplotlib jupyter
```
Additionally, install the TensorFlow Object Detection API
```bash
git clone https://github.com/tensorflow/models.git
cd models/research
protoc object_detection/protos/*.proto --python_out=.
cp object_detection/packages/tf2/setup.py .
pip install .
```

## Running the Code

Clone this repository:
```bash
git clone https://github.com/PubDe/ZeaMays-Seed-Detection-LowLight.git
cd ZeaMays-Seed-Detection-LowLight
```
To run the seed detection script:
```bash
python seed-detection.py
```

### Using Jupyter Notebook
To use the provided Jupyter Notebook for object detection:

1. Start Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
2. Open the provided ```seed-detection.ipynb``` file and run the cells step by step

## Sample Predictions
![predict_img-1](https://github.com/PubDe/ZeaMays-Seed-Detection-LowLight/blob/main/seed_count-predict_img%20(3).jpg?raw=true)
## 
![predict_img-2](https://github.com/PubDe/ZeaMays-Seed-Detection-LowLight/blob/main/seed_count-predict_img%20(1).jpg?raw=true)
## 
![predict_img-3](https://github.com/PubDe/ZeaMays-Seed-Detection-LowLight/blob/main/seed_count-predict_img%20(7).jpg?raw=true)

## Model Configuration

To train a custom model, configure the ```pipeline.config``` file to match your dataset and environment settings. Ensure the paths to your dataset and pre-trained model are correct.
