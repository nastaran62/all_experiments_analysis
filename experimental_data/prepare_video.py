import os
import cv2

def prepare_data(input_path, output_path, baseline=5):
    all_participants = os.listdir(input_path)
    all_participants.sort()
    for participant in all_participants:
        all_video_path = os.path.join(os.path.join(input_path, participant), "video")
        video_files = os.listdir(all_video_path)
        video_files.sort()
        for video in video_files:
            print(video)
            video_path = os.path.join(all_video_path, video)
            vidcap = cv2.VideoCapture(video_path)
            fps, frames = vidcap.get(cv2.CAP_PROP_FPS), vidcap.get(cv2.CAP_PROP_FRAME_COUNT)
            all_frames = []
            while(1):
                success, frame = vidcap.read()
                if not success:
                    break
                all_frames.append(frame)
            fps = int(fps)
            all_frames = all_frames[fps*baseline:]


            part_length = fps*60
            part_count = int(len(all_frames) / part_length)
            start = 0
            end = part_length
            i = 0
            while end < len(all_frames):
                folder = os.path.join(output_path, participant)
                if not os.path.exists(folder):
                    os.mkdir(folder)
                file_name = \
                    "{0}/{1}-{2}.avi".format(folder,
                                             video[:-4], str(i).zfill(2))
                print(file_name)

                part = all_frames[start:end]
                coded = cv2.VideoWriter_fourcc(*'XVID')
                writer = cv2.VideoWriter(file_name,
                                         coded,
                                         fps,
                                         (1280, 720))
                for frame in part:
                    writer.write(frame)
                writer.release()

                start = end
                end = end + part_length
                i += 1

prepare_data("/home/zsaf419/backup-data/collected_data/conversation",
             "/home/zsaf419/Documents/project/all_experiments_analysis/experimental_data/exp1_1/prepared_video")
