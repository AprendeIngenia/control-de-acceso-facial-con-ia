import numpy as np
from typing import Any

from numpy import ndarray

from process.face_processing.face_utils import FaceUtils
from process.database.config import DataBasePaths


class FacialLogIn:
    def __init__(self):
        self.face_utilities = FaceUtils()
        self.database = DataBasePaths()

        self.face_comparison = False
        self.matcher = None

    def process(self, face_image: np.ndarray) -> tuple[ndarray, None, str] | tuple[ndarray, bool, str] | tuple[
        ndarray, bool | Any, str]:
        # step 1: check face detection
        check_face_detect, face_info, face_save = self.face_utilities.check_face(face_image)
        if check_face_detect is False:
            self.face_comparison = False
            self.matcher = None
            return face_image, self.matcher, '¡No face detect!'

        # step 2: face mesh
        check_face_mesh, face_mesh_info = self.face_utilities.face_mesh(face_image)
        if check_face_mesh is False:
            return face_image, self.matcher, '¡No face mesh detect!'

        # step 3: extract face mesh
        face_mesh_points_list = self.face_utilities.extract_face_mesh(face_image, face_mesh_info)

        # step 4: check face center
        check_face_center = self.face_utilities.check_face_center(face_mesh_points_list)

        # step 5: show state
        self.face_utilities.show_state_login(face_image, state=self.matcher)

        if check_face_center:
            # step 5: extract face info
            # bbox & key points
            face_bbox = self.face_utilities.extract_face_bbox(face_image, face_info)
            face_points = self.face_utilities.extract_face_points(face_image, face_info)

            # step 6: face aligned
            face_aligned = self.face_utilities.face_alignment(face_save, face_points)

            # step 7: face crop
            face_crop = self.face_utilities.face_crop(face_aligned, face_bbox)

            # step 8: read face database
            faces_database, names_database, info = self.face_utilities.read_face_database(self.database.faces)

            if len(faces_database) != 0 and not self.face_comparison and self.matcher is None:
                self.face_comparison = True
                # step 9: compare faces
                self.matcher, user_name = self.face_utilities.face_matching(face_crop, faces_database, names_database)

                if self.matcher:
                    # step 10: save date & time
                    self.face_utilities.user_register(user_name, self.database.users)
                    return face_image, self.matcher, 'Approved user access!'
                else:
                    return face_image, self.matcher, 'User no approved!'
            else:
                return face_image, self.matcher, 'Empty database!'

        return face_image, self.matcher, 'No center face!'
