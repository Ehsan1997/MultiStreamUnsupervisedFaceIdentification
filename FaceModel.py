import torch


class FaceModel:
    def __init__(self,detector, embedder, sim=None, thresh=0.5):
        self.detector = detector
        self.embedder = embedder
        self.sim = sim if sim else torch.nn.CosineSimilarity()
        self.thresh = thresh

    def detect_faces(self, frame):
        self.detector.eval()
        return self.detector(frame)

    def extract_facial_features(self, faces):
        self.embedder.eval()
        with torch.no_grad():
            return self.embedder(faces)
