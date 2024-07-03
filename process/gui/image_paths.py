from pydantic import BaseModel

from process.gui.setup.images.face_capture_button import face_capture_image_path
from process.gui.setup.images.gui_init_image import gui_init_image_path
from process.gui.setup.images.gui_signup_image import gui_signup_image_path
from process.gui.setup.images.login_button import login_button_image_path
from process.gui.setup.images.signup_button import signup_button_image_path


class ImagePaths(BaseModel):
    # main images
    init_img: str = gui_init_image_path
    login_img: str = login_button_image_path
    signup_img: str = signup_button_image_path

    # secondary windows
    gui_signup_img: str = gui_signup_image_path
    register_img: str = face_capture_image_path
