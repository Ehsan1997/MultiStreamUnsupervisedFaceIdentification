from collections import Counter, defaultdict
from pathlib import Path
import pickle
from time import strftime, gmtime

import av
import torch
from PIL import Image


class Channel:
    def __init__(self, name, url, frame_retriever, face_model, root_path, stats=None, db=None):
        self.url = url
        self.db = db
        self.name = name
        self.frame_retriever = frame_retriever(self.url)
        self.face_model = face_model
        self.stats = stats if stats else {
            'person_frequency': Counter(),
            'frame_count': 0,
            'person_to_time': defaultdict(Counter),
            'time_to_person': defaultdict(Counter)
        }
        # Create Folders for required Directories.
        self.root_path = Path(root_path)
        self.path_dict = {"frames": self.root_path/Path("Frames"),
                          "stats": self.root_path/Path("Stats"),
                          "faces": self.root_path/Path("Faces")}
        for key in self.path_dict:
            self.path_dict[key].mkdir(exist_ok=True, parents=True)

    def get_frame(self):
        # Return frame
        try:
            frame = next(self.frame_retriever)
            self.stats["frame_count"] += 1
        except StopIteration:
            frame = None
        return frame

    def assign_id(self, embedding):
        if self.db is None:
            self.db = embedding
            return 0
        sim_scores = self.face_model.sim(embedding, self.db)
        person_id = torch.argmax(sim_scores).item()
        score = sim_scores[person_id].item()
        if score > self.face_model.thresh:
            return person_id
        # New Person is detected.
        else:
            self.db = torch.cat((self.db, embedding))
            return self.db.shape[0] - 1

    def update_stats(self, person_id):
        self.stats["person_frequency"][person_id] += 1
        # Get Timestamp for the current time.
        timestamp = strftime("%Y%m%d%H%M", gmtime())
        timestamp = timestamp[:-1]
        self.stats["person_to_time"][person_id][timestamp] += 1
        self.stats["time_to_person"][timestamp][person_id] += 1

    def save_frame(self, frame):
        Image.fromarray(frame).save(
            self.path_dict['frames']/Path(f'{self.stats["frame_count"]-1}.png')
        )

    def save_face(self, face, person_id):
        face_path = self.path_dict["faces"]/Path(str(person_id))
        face_path.mkdir(parents=True, exist_ok=True)
        denormalized_image = ((face.permute(1, 2, 0).cpu().numpy() * 128.0) + 127.5).astype('uint8')
        Image.fromarray(denormalized_image).save(
            face_path/Path(f'{self.stats["person_frequency"][person_id]-1}.png')
        )

    def save_stats(self):
        with open(self.path_dict['stats']/Path("stats.pkl"), 'wb') as f:
            pickle.dump(self.stats, f, -1)
