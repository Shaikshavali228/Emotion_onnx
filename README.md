Installation
conda create -n ONNX python=3.8
conda activate ONNX
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
pip install onnxruntime-gpu==1.14.0
pip install opencv-python==4.5.5.64
Features:
Real-time face detection using ONNX model.
Emotion recognition from detected faces using another ONNX model.
Visualization of emotion scores on the video stream.
Download the pre-trained ONNX models:
Place the downloaded models in the weights directory
Download	weight
WebCam Inference
$ python demo.py
Note
This repo supports only inference, see reference for more details
