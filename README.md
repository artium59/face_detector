# Face Detection 

![Detection Example](https://i.imgur.com/B2pREOJ.jpg)

## Usage

```
usage: python face_detection.py [--type] [--input] [--output]

optional arguments:
  -h, --help       show this help message and exit
  --type TYPE      choose picture(p) or video and webcam(v)
  --input INPUT    input path
  --output OUTPUT  output path (if 0, use webcam.)
```

---
## Some issues to know
1. The test environment is
    - Python 3.7.3
    - OpenCV 3.4.2
    - model.py     : azure cognitiveservice vision face 0.4.0
    - free_model.py: cognitive face 1.5.0

2. If you use this, you should get a subscription key in [Face API](https://azure.microsoft.com/en-us/services/cognitive-services/face/).

3. If you use Guest Account(7-day trial), you use free_model.py and change [this](https://github.com/artium59/face_detector/blob/d24174f4707c9ba5508f86ac445e1b3a52b3b9fb/face_detecion.py#L2) model to free_model.

4. You should change KEY and BASE_URL / ENDPOINT_STRING.
