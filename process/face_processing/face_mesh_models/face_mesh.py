import numpy as np
import mediapipe as mp
import cv2
from typing import Any, List, Tuple


class FaceMeshMediapipe:
    def __init__(self):
        # mediapipe
        self.mp_draw = mp.solutions.drawing_utils
        self.config_draw = self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=1)

        self.face_mesh_object = mp.solutions.face_mesh
        self.face_mesh_mp = self.face_mesh_object.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                           refine_landmarks=False, min_detection_confidence=0.6,
                                                           min_tracking_confidence=0.6)

        self.mesh_points = None
        # face points
        # right parietal
        self.rp_x: int = 0
        self.rp_y: int = 0
        # left parietal
        self.lp_x: int = 0
        self.lp_y: int = 0
        # right eyebrow
        self.re_x: int = 0
        self.re_y: int = 0
        # left eyebrow
        self.le_x: int = 0
        self.le_y: int = 0

    def face_mesh_mediapipe(self, face_image: np.ndarray) -> Tuple[bool, Any]:
        rgb_image = face_image.copy()
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        face_mesh = self.face_mesh_mp.process(rgb_image)
        if face_mesh.multi_face_landmarks is None:
            return False, face_mesh
        else:
            return True, face_mesh

    def extract_face_mesh_points(self, face_image: np.ndarray, face_mesh_info: Any, viz: bool) -> List[List[int]]:
        height, width, _ = face_image.shape
        self.mesh_points = []
        for face_mesh in face_mesh_info.multi_face_landmarks:
            for i, points in enumerate(face_mesh.landmark):
                x, y = int(points.x * width), int(points.y * height)
                self.mesh_points.append([i, x, y])

            if viz:
                self.mp_draw.draw_landmarks(face_image, face_mesh, self.face_mesh_object.FACEMESH_TESSELATION,
                                            self.config_draw, self.config_draw)

        return self.mesh_points

    def check_face_center(self, face_points: List[List[int]]) -> bool:
        if len(face_points) == 468:
            self.rp_x, self.rp_y = face_points[139][1:]
            self.lp_x, self.lp_y = face_points[368][1:]
            self.re_x, self.re_y = face_points[70][1:]
            self.le_x, self.le_y = face_points[300][1:]

            if self.re_x > self.rp_x and self.le_x < self.lp_x:
                return True
            else:
                return False

    def config_color(self, color: Tuple[int, int, int]):
        self.config_draw = self.mp_draw.DrawingSpec(color=color, thickness=1, circle_radius=1)
