import colorsys
import operator
from timeit import default_timer as timer

import cognitive_face as CF
import numpy as np
from PIL import Image, ImageFont, ImageDraw

# Replace with a valid subscription key (keeping the quotes in place).
KEY = '<Subscription Key>'
CF.Key.set(KEY)

# Replace with your regional Base URL
BASE_URL = 'https://westus.api.cognitive.microsoft.com/face/v1.0/'
CF.BaseUrl.set(BASE_URL)


def make_colors():
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / 20, 1., 1.) for x in range(20)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    return colors


# Convert width height to a point in a rectangle
def get_rectangle(faceDictionary):
    rect = faceDictionary['faceRectangle']
    left = rect['left']
    top = rect['top']
    bottom = left + rect['height']
    right = top + rect['width']

    return ((left, top), (bottom, right))


def draw_face(image_path, image, colors):
    """
    'faceAttributes': {
      'emotion': {
        'anger': 0.0,
        'contempt': 0.005,
        'disgust': 0.0,
        'fear': 0.0,
        'happiness': 0.372,
        'neutral': 0.623,
        'sadness': 0.0,
        'surprise': 0.0
      }
    }
    """
    faces = CF.face.detect(image_path, attributes='emotion')
    """
    for face in faces:
        print(type(face['faceAttributes']['emotion']), face['faceAttributes']['emotion'],
              max(face['faceAttributes']['emotion'].items(), key=operator.itemgetter(1))[0])
    """

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

    for c, face in enumerate(faces):
        label = max(face['faceAttributes']['emotion'].items(), key=operator.itemgetter(1))[0]
        label = '{} {}'.format(label, face['faceAttributes']['emotion'][label])
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        (left, top), _ = get_rectangle(face)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        draw.rectangle(get_rectangle(face), outline=colors[c])
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image


def detect_picture(image_path, output_path=""):
    # Detect a face in an image that contains faces
    colors = make_colors()

    img = Image.open(image_path)
    img = draw_face(image_path, img, colors)
    img.show()

    if output_path != "":
        img.save(output_path)


def detect_video(video_path, output_path=""):
    import cv2

    if video_path == '0': video_path = int(video_path)
    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    colors = make_colors()

    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()

    while True:
        return_value, frame = vid.read()
        image = Image.fromarray(frame)
        if curr_fps % 5 == 0:
            image.save('./output/temperary.jpg')
            image = draw_face('./output/temperary.jpg', image, colors)
        result = np.asarray(image)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1

        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0

        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        if isOutput:
            out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
