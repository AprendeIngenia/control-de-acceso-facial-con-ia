import face_recognition as fr
from deepface import DeepFace
from typing import Tuple
import cv2
import numpy as np
import json


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
        pass

    def face_matching_face_recognition_model(self, id_face: np.ndarray, user_face: np.ndarray) -> Tuple[bool, float]:
        id_face = cv2.cvtColor(id_face, cv2.COLOR_BGR2RGB)
        user_face = cv2.cvtColor(user_face, cv2.COLOR_BGR2RGB)

        id_face_loc = [(0, id_face.shape[0], id_face.shape[1], 0)]
        user_face_loc = [(0, user_face.shape[0], user_face.shape[1], 0)]

        id_face_encoding = fr.face_encodings(id_face, known_face_locations=id_face_loc)[0]
        user_face_encoding = fr.face_encodings(user_face, known_face_locations=user_face_loc)

        matching = fr.compare_faces(id_face_encoding, user_face_encoding, tolerance=0.55)
        distance = fr.face_distance(id_face_encoding, user_face_encoding)

        return matching[0], distance[0]

    def face_matching_vgg_model(self, id_face: np.ndarray, user_face: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=id_face, img2_path=user_face, model_name=self.models[0])
            matching, distance = result['verified'], result['distance']
            # print(json.dumps(result, indent=2))
            return matching, distance
        except:
            return False, 0.0

    def face_matching_facenet_model(self, id_face: np.ndarray, user_face: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=id_face, img2_path=user_face, model_name=self.models[1])
            matching, distance = result['verified'], result['distance']
            #print(json.dumps(result, indent=2))
            return matching, distance
        except:
            return False, 0.0

    def face_matching_facenet512_model(self, id_face: np.ndarray, user_face: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=id_face, img2_path=user_face, model_name=self.models[2])
            matching, distance = result['verified'], result['distance']
            #print(json.dumps(result, indent=2))
            return matching, distance
        except:
            return False, 0.0

    def face_matching_openface_model(self, id_face: np.ndarray, user_face: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=id_face, img2_path=user_face, model_name=self.models[3])
            matching, distance = result['verified'], result['distance']
            #print(json.dumps(result, indent=2))
            return matching, distance
        except:
            return False, 0.0

    def face_matching_deepface_model(self, id_face: np.ndarray, user_face: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=id_face, img2_path=user_face, model_name=self.models[4])
            matching, distance = result['verified'], result['distance']
            #print(json.dumps(result, indent=2))
            return matching, distance
        except:
            return False, 0.0

    def face_matching_deepID_model(self, id_face: np.ndarray, user_face: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=id_face, img2_path=user_face, model_name=self.models[5])
            matching, distance = result['verified'], result['distance']
            #print(json.dumps(result, indent=2))
            return matching, distance
        except:
            return False, 0.0

    def face_matching_arcface_model(self, id_face: np.ndarray, user_face: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=id_face, img2_path=user_face, model_name=self.models[6])
            matching, distance = result['verified'], result['distance']
            #print(json.dumps(result, indent=2))
            return matching, distance
        except:
            return False, 0.0

    def face_matching_sface_model(self, id_face: np.ndarray, user_face: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=id_face, img2_path=user_face, model_name=self.models[7])
            matching, distance = result['verified'], result['distance']
            #print(json.dumps(result, indent=2))
            return matching, distance
        except:
            return False, 0.0

    def face_matching_dlib_model(self, id_face: np.ndarray, user_face: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=id_face, img2_path=user_face, model_name=self.models[7])
            matching, distance = result['verified'], result['distance']
            #print(json.dumps(result, indent=2))
            return matching, distance
        except:
            return False, 0.0

    def face_matching_ghostfacenet_model(self, id_face: np.ndarray, user_face: np.ndarray) -> Tuple[bool, float]:
        try:
            result = DeepFace.verify(img1_path=id_face, img2_path=user_face, model_name=self.models[8])
            matching, distance = result['verified'], result['distance']
            #print(json.dumps(result, indent=2))
            return matching, distance
        except:
            return False, 0.0
