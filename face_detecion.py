import argparse
from model import detect_video, detect_picture

FLAGS = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--type", nargs='?', type=str, required=True, default='video',
        help="picture(p) or video(v)"
    )

    parser.add_argument(
        "--input", nargs='?', type=str, required=True, default='./path2your_video',
        help="input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help="[Optional] output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.type == 'video' or FLAGS.type == 'v':
        detect_video(FLAGS.input, FLAGS.output)
    elif FLAGS.type == 'picture' or FLAGS.type == 'p':
        detect_picture(FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
