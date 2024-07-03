import face_recognition as fr
from deepface import DeepFace
from typing import Tuple
import cv2
import numpy as np


class FaceMatcherModels:
    def __init__(self):
        self.models = [
            "VGG-Face",
            "Facenet",
            "Facenet512",
            "OpenFace",
            "DeepFace",
            "DeepID",
            "ArcFace",
            "Dlib",
            "SFace",
            "GhostFaceNet",
        ]

    def face_matching_face_recognition_model(self, face_1: np.ndarray, face_2: np.ndarray) -> Tuple[bool, float]:
        face_1 = cv2.cvtColor(face_1, cv2.COLOR_BGR2RGB)
        face_2 = cv2.cvtColor(face_2, cv2.COLOR_BGR2RGB)

        face_loc_1 = [(0, face_1.shape[0], face_1.shape[1], 0)]
        face_loc_2 = [(0, face_2.shape[0], face_2.shape[1], 0)]

        face_1_encoding = fr.face_encodings(face_1, known_face_locations=face_loc_1)[0]
        face_2_encoding = fr.face_encodings(face_2, known_face_locations=face_loc_2)

        matching = fr.compare_faces(face_1_encoding, face_2_encoding, tolerance=0.55)
        distance = fr.face_distance(face_1_encoding, face_2_encoding)

        return matching[0], distance[0]

    def face_matching_vgg_model(self, face_1: np.ndarray, face_2: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=face_1, img2_path=face_2, model_name=self.models[0])
            matching, distance = result['verified'], result['distance']
            return matching, distance
        except:
            return False, 0.0

    def face_matching_facenet_model(self, face_1: np.ndarray, face_2: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=face_1, img2_path=face_2, model_name=self.models[1])
            matching, distance = result['verified'], result['distance']
            return matching, distance
        except:
            return False, 0.0

    def face_matching_facenet512_model(self, face_1: np.ndarray, face_2: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=face_1, img2_path=face_2, model_name=self.models[2])
            matching, distance = result['verified'], result['distance']
            return matching, distance
        except:
            return False, 0.0

    def face_matching_openface_model(self, face_1: np.ndarray, face_2: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=face_1, img2_path=face_2, model_name=self.models[3])
            matching, distance = result['verified'], result['distance']
            return matching, distance
        except:
            return False, 0.0

    def face_matching_deepface_model(self, face_1: np.ndarray, face_2: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=face_1, img2_path=face_2, model_name=self.models[4])
            matching, distance = result['verified'], result['distance']
            return matching, distance
        except:
            return False, 0.0

    def face_matching_deepid_model(self, face_1: np.ndarray, face_2: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=face_1, img2_path=face_2, model_name=self.models[5])
            matching, distance = result['verified'], result['distance']
            return matching, distance
        except:
            return False, 0.0

    def face_matching_arcface_model(self, face_1: np.ndarray, face_2: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=face_1, img2_path=face_2, model_name=self.models[6])
            matching, distance = result['verified'], result['distance']
            return matching, distance
        except:
            return False, 0.0

    def face_matching_dlib_model(self, face_1: np.ndarray, face_2: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=face_1, img2_path=face_2, model_name=self.models[7])
            matching, distance = result['verified'], result['distance']
            return matching, distance
        except:
            return False, 0.0

    def face_matching_sface_model(self, face_1: np.ndarray, face_2: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=face_1, img2_path=face_2, model_name=self.models[8])
            matching, distance = result['verified'], result['distance']
            return matching, distance
        except:
            return False, 0.0

    def face_matching_ghostfacenet_model(self, face_1: np.ndarray, face_2: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=face_1, img2_path=face_2, model_name=self.models[9])
            matching, distance = result['verified'], result['distance']
            return matching, distance
        except:
            return False, 0.0
