# Convolutional-neural-network-for-image-recognition
This convolutional neural network uses the keras and the tensorflow backend.  
It has been trained on 8000 images of cats and dogs. Given an image of a cat or a dog, 
it can give an accurate prediction of what the object in the image is.


- create venv
```
python3.6 -m pip install -U pip setuptools wheel
python3.6 -m venv env
. env/bin/activate
pip install -e .
```
- build the model
```
python src/cnn.py --build
```
- test the model
```
python src/cnn.py --test
```
- if on mac, you may need to `brew install tcl-tk` or `sudo apt-get install python3-tk` on ubuntu


### TODO
- [ ] add REST interface for making request
- [ ] better accuracy
- [ ] try different algorithm from keras
- [ ] try LSTM model for Image classification