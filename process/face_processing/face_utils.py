import os
import numpy as np
import cv2
import datetime
from typing import List, Tuple, Any
from process.face_processing.face_detect_models.face_detect import FaceDetectMediapipe
from process.face_processing.face_mesh_models.face_mesh import FaceMeshMediapipe
from process.face_processing.face_matcher_models.face_matcher import FaceMatcherModels


class FaceUtils:
    def __init__(self):
        # face detect
        self.face_detector = FaceDetectMediapipe()
        # face mesh
        self.mesh_detector = FaceMeshMediapipe()
        # face matcher
        self.face_matcher = FaceMatcherModels()

        # variables
        self.angle = None
        self.face_db = []
        self.face_names = []
        self.distance: float = 0.0
        self.matching: bool = False
        self.user_registered = False

    # detect
    def check_face(self, face_image: np.ndarray) -> Tuple[bool, Any, np.ndarray]:
        face_save = face_image.copy()
        check_face, face_info = self.face_detector.face_detect_mediapipe(face_image)
        return check_face, face_info, face_save

    def extract_face_bbox(self, face_image: np.ndarray, face_info: Any):
        h_img, w_img, _ = face_image.shape
        bbox = self.face_detector.extract_face_bbox_mediapipe(w_img, h_img, face_info)
        return bbox

    def extract_face_points(self, face_image: np.ndarray, face_info: Any):
        h_img, w_img, _ = face_image.shape
        face_points = self.face_detector.extract_face_points_mediapipe(h_img, w_img, face_info)
        return face_points

    # face mesh
    def face_mesh(self, face_image: np.ndarray) -> Tuple[bool, Any]:
        check_face_mesh, face_mesh_info = self.mesh_detector.face_mesh_mediapipe(face_image)
        return check_face_mesh, face_mesh_info

    def extract_face_mesh(self, face_image: np.ndarray, face_mesh_info: Any) -> List[List[int]]:
        face_mesh_points_list = self.mesh_detector.extract_face_mesh_points(face_image, face_mesh_info, viz=True)
        return face_mesh_points_list

    def check_face_center(self, face_points: List[List[int]]) -> bool:
        check_face_center = self.mesh_detector.check_face_center(face_points)
        return check_face_center

    # crop
    def face_crop(self, face_image: np.ndarray, face_bbox: List[int]) -> np.ndarray:
        h, w, _ = face_image.shape
        offset_x, offset_y = int(w * 0.025), int(h * 0.025)
        xi, yi, xf, yf = face_bbox
        xi, yi, xf, yf = xi - offset_x, yi - (offset_y*4), xf + offset_x, yf
        return face_image[yi:yf, xi:xf]

    # save
    def save_face(self, face_crop: np.ndarray, user_code: str, path: str):
        if len(face_crop) != 0:
            face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{path}/{user_code}.png", face_crop)
            return True

        else:
            return False

    # draw
    def show_state_signup(self, face_image: np.ndarray, state: bool):
        if state:
            text = 'Saving face, wait three seconds please'
            size_text = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = size_text[0], size_text[1]
            cv2.rectangle(face_image, (370, 650 - dim[1]-baseline), (370 + dim[0], 650 + baseline), (0, 0, 0), cv2.FILLED)
            cv2.putText(face_image, text, (370, 650-5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 1)
            self.mesh_detector.config_color((0, 255, 0))

        else:
            text = 'Face processing, see the camera please!'
            size_text = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = size_text[0], size_text[1]
            cv2.rectangle(face_image, (370, 650 - dim[1] - baseline), (370 + dim[0], 650 + baseline), (0, 0, 0),
                          cv2.FILLED)
            cv2.putText(face_image, text, (370, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 0), 1)
            self.mesh_detector.config_color((255, 0, 0))

    def show_state_login(self, face_image: np.ndarray, state: bool):
        if state:
            text = 'Approved face, come in please!'
            size_text = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = size_text[0], size_text[1]
            cv2.rectangle(face_image, (370, 650 - dim[1] - baseline), (370 + dim[0], 650 + baseline), (0, 0, 0),
                          cv2.FILLED)
            cv2.putText(face_image, text, (370, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 255, 0), 1)
            self.mesh_detector.config_color((0, 255, 0))

        elif state is None:
            text = 'Comparing faces, see the camera and wait 3 seconds please!'
            size_text = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = size_text[0], size_text[1]
            cv2.rectangle(face_image, (250, 650 - dim[1] - baseline), (250 + dim[0], 650 + baseline), (0, 0, 0),
                          cv2.FILLED)
            cv2.putText(face_image, text, (250, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 0), 1)
            self.mesh_detector.config_color((255, 255, 0))

        elif state is False:
            text = 'Face no approved, please register!'
            size_text = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)
            dim, baseline = size_text[0], size_text[1]
            cv2.rectangle(face_image, (370, 650 - dim[1] - baseline), (370 + dim[0], 650 + baseline), (0, 0, 0),
                          cv2.FILLED)
            cv2.putText(face_image, text, (370, 650 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 0, 0), 1)
            self.mesh_detector.config_color((255, 0, 0))

    def read_face_database(self, database_path: str) -> Tuple[List[np.ndarray], List[str], str]:
        self.face_db: List[np.ndarray] = []
        self.face_names: List[str] = []

        for file in os.listdir(database_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(database_path, file)
                img_read = cv2.imread(img_path)
                if img_read is not None:
                    self.face_db.append(img_read)
                    self.face_names.append(os.path.splitext(file)[0])

        return self.face_db, self.face_names, f'Comparing {len(self.face_db)} faces!'

    def face_matching(self, current_face: np.ndarray, face_db: List[np.ndarray], name_db: List[str]) -> Tuple[bool, str]:
        user_name: str = ''
        current_face = cv2.cvtColor(current_face, cv2.COLOR_RGB2BGR)
        for idx, face_img in enumerate(face_db):
            self.matching, self.distance = self.face_matcher.face_matching_sface_model(current_face, face_img)
            print(f'validating face with: {name_db[idx]}')
            print(f'matching: {self.matching} distance: {self.distance}')
            if self.matching:
                user_name = name_db[idx]
                return self.matching, user_name
        return False, 'Face unknown'

    def user_check_in(self, user_name: str, user_path: str):
        if not self.user_registered:
            now = datetime.datetime.now()
            date_time = now.strftime("%Y-%m-%d %H:%M:%S")
            user_file_path = os.path.join(user_path, f"{user_name}.txt")
            with open(user_file_path, "a")as user_file:
                user_file.write(f'\nAccess granted at: {date_time}\n')

            self.user_registered = True
