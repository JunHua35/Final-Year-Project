import cv2
import os
from mtcnn.mtcnn import MTCNN

image_size = 256


def video_to_frames(data_dir, frames_to_extract):

    """data_dir: videos input path"""

    if frames_to_extract < 1:
        raise Exception("Choose at least 1 frame to extract from the video")

    files = os.listdir(data_dir)

    if data_dir.find('videos') != -1:
        file1 = open(
            #the text file should include the name of the videos to be extracted
            '/media/monash/SSD/MCS1/MesoNet/faceforensics_Dataset/testing_extract_2.txt',
            'r')
        files = [line.rstrip() for line in file1]

    vid_list = [file for file in files]
    vid_list.sort()
    vid_count = 0
    for vid in vid_list:
        counter = 0
        inner_counter = 0

        # total process video limit
        if vid_count > 500:
            break

        # to skip processed video & process with the rest (control RAM)
        elif vid_count < 0:
            vid_count += 1
            continue
        else:
            notFrameEnd = True
            sec = 0
            frameRate = 0.5  # //it will capture image in each 0.1 second
            while notFrameEnd:


                # control the number of frames extract per video
                if counter >= frames_to_extract:
                    break
                else:

                    # inner counter control subsequent valid frame process (filter bias frame)
                    if inner_counter > 0:
                        sec = sec + frameRate
                        sec = round(sec, 2)

                    vidcap = cv2.VideoCapture(os.path.join(data_dir, vid))
                    frameId = vidcap.get(1)
                    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
                    hasFrames, image = vidcap.read()
                    notFrameEnd = hasFrames

                    if hasFrames:
                        detector = MTCNN()
                        face_rects = detector.detect_faces(image)
                        if len(face_rects) > 0:
                            x1, y1, width, height = face_rects[0]['box']
                            x2, y2 = x1 + width, y1 + height
                            if x1 >= 0 and y1 >= 0:
                                image = image[y1:y2, x1:x2]

                            # output path 
                            output = '/media/monash/SSD/MCS1/MesoNet/faceforensics_Dataset/train_dataset/train_images/manipulated/'

                            output_path = os.path.join(os.path.expanduser('~'), output)
                            if not os.path.exists(output_path):
                                os.makedirs(output_path)
                            try:

                                image = cv2.resize(image, (image_size, image_size))
                                cv2.imwrite(output_path + vid.split('.')[0] + '_' + str(counter) + '.png', image)

                                if not cv2.imwrite(output_path + vid.split('.')[0] + '_' + str(counter) + '.png', image):
                                    raise Exception("Could not write image")
                            except Exception as e:
                                print(str(e))

                            print(counter)
                            counter += 1
                            inner_counter += 1
                        else:
                            inner_counter += 1

            vid_count += 1
            print("processing vidio: ", vid_count)
            
#input path that includes the video
video_to_frames('/media/monash/SSD/MCS1/MesoNet/faceforensics_Dataset/manipulated_sequences/Face2Face/c23/videos', 10)
