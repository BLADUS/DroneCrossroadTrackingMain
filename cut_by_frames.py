import cv2
import os


def get_frames(video_path, out_folder, frame_interval=40):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True
    while success:
        success, frame = video.read()
        if frame_count % frame_interval == 0:
            output_path = os.path.join(out_folder, f"frame_{0 + frame_count // frame_interval}.jpg")
            cv2.imwrite(output_path, frame)
        frame_count += 1
    video.release()


def main():
    pwd = os.getcwd()
    video_path = f'{pwd}\\footage\\crossroad12.mp4'
    output_folder = f'{pwd}\\test_frames\\'
    get_frames(video_path, output_folder)


if __name__ == '__main__':
    main()
