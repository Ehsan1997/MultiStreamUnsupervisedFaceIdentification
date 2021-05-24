# TODO: Live Stream Capability.
import streamlink

from Channel import Channel
from FaceModel import FaceModel
from facenet_pytorch import MTCNN, InceptionResnetV1
import av


def frame_retriever_base(container, n_skip_frame=30):
    for i, frame in enumerate(container.decode(video=0)):
        if i % n_skip_frame == 0:
            yield frame.to_ndarray(format='rgb24')


def get_frame_from_vid(vid_path, n_skip_frame=30):
    container = av.open(vid_path)
    return frame_retriever_base(container, n_skip_frame)


def get_frame_from_vid_live(vid_url, n_skip_frame=30):
    vid_url = streamlink.streams(vid_url)['best'].url
    container = av.open(vid_url, format='segment')
    return frame_retriever_base(container, n_skip_frame)


def recognize_faces(channel):
    frame = channel.get_frame()
    if frame is None:
        return
    channel.save_frame(frame)
    face_tensor = channel.face_model.detect_faces(frame)
    if face_tensor is None:
        return
    embedding_tensor = channel.face_model.extract_facial_features(face_tensor)
    # Unsqueeze to change from (n,512) -> (n, 1, 512)
    embedding_tensor = embedding_tensor.unsqueeze(1)
    for i in range(len(face_tensor)):
        person_id = channel.assign_id(embedding_tensor[i])
        channel.update_person_frequency(person_id)
        channel.save_face(face_tensor[i], person_id)
    channel.save_stats()


face_model = FaceModel(
    MTCNN(
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            keep_all=True
        ),
    InceptionResnetV1(pretrained='vggface2')
)
# vid_path = r'C:\Users\Ehsan\PycharmProjects\live_face_rec_dashboard\SampleVideo_CapitalTalk.mp4'
# vid_path1 = r'C:\Users\Ehsan\PycharmProjects\live_face_rec_dashboard\SampleVid2-KamranKhan.mp4'

bol_url = 'https://www.youtube.com/watch?v=VEFUISG8ooc'
ninetytwo_url = 'https://www.youtube.com/watch?v=zWj87fKufXg'

channel1 = Channel("bol", bol_url, get_frame_from_vid_live, face_model, "Dump/Bol")
channel2 = Channel("92-news", ninetytwo_url, get_frame_from_vid_live, face_model, 'Dump/92_News')

channels = [channel1, channel2]

for _ in range(100):
    for channel_ in channels:
        recognize_faces(channel_)
        print(channel_.name, ":", channel_.stats)