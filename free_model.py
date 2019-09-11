import colorsys
from timeit import default_timer as timer

import numpy as np
from PIL import Image, ImageFont, ImageDraw
from azure.cognitiveservices.vision.face import FaceClient
from azure.cognitiveservices.vision.face.models import FaceAttributeType
from msrest.authentication import CognitiveServicesCredentials

# Set the FACE_SUBSCRIPTION_KEY environment variable with your key as the value.
# This key will serve all examples in this document.
KEY = '<Subscription Key>'

# Set the API endpoint for your Face subscription.
# You may need to change the first part ("westus") to match your subscription
ENDPOINT_STRING = "westcentralus"

ENDPOINT = 'https://{}.api.cognitive.microsoft.com/'.format(ENDPOINT_STRING)

# Create an authenticated FaceClient.
face_client = FaceClient(ENDPOINT, CognitiveServicesCredentials(KEY))


def find_max_emotion(emotions):
    lines = '{}'.format(emotions)
    lines = lines.split(',')
    lines = [x for x in lines if len(x) > 0]  # get read of the empty lines
    lines = [x for x in lines if x[0] != '#']  # get rid of comments
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces

    max_emotion = ''
    max_percent = float(0)
    for line in lines[1:]:
        key, value = line.split(':')
        key = key.rstrip().lstrip()
        value = value.rstrip().lstrip()
        if value[-1] == '}':
            value = value[:-1]
        if max_percent < float(value):
            max_emotion = key[1:-1]
            max_percent = float(value)

    return max_emotion, max_percent


def make_colors():
    # Generate colors for drawing bounding boxes.
    hsv_tuples = [(x / 20, 1., 1.)
                  for x in range(20)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(
        map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
            colors))
    np.random.seed(10101)  # Fixed seed for consistent colors across runs.
    np.random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    np.random.seed(None)  # Reset seed to default.

    return colors


# Convert width height to a point in a rectangle
def get_rectangle(face_info):
    rect = face_info.face_rectangle
    left = rect.left
    top = rect.top
    bottom = left + rect.height
    right = top + rect.width

    return ((left, top), (bottom, right))


def draw_face(image_path, image, colors):
    img = open(image_path, 'r+b')
    print(img)
    # Line 707
    faces = face_client.face.detect_with_stream(img, return_face_attributes=FaceAttributeType.emotion)

    font = ImageFont.truetype(font='font/FiraMono-Medium.otf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))

    for c, face in enumerate(faces):
        label = '{}'.format(find_max_emotion(face.face_attributes.emotion))
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


def detect_video(video_path, output_path=""):
    import cv2

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

        if curr_fps % 180 == 0:
            image.save('./output/temperary.jpg')
            image = draw_face('./output/temperary.jpg', image, colors)
            curr_fps = 0
        result = np.asarray(image)

        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1

        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)

        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)

        if isOutput:
            out.write(result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def detect_picture(image_path, output_path=""):
    # Detect a face in an image that contains faces
    colors = make_colors()

    img = Image.open(image_path)
    img = draw_face(image_path, img, colors)
    img.show()

    if output_path != "":
        img.save(output_path)
