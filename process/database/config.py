from pydantic import BaseModel
from process.database.faces_path import faces_path
from process.database.users_path import (users_path, users_check_path)


class DataBasePaths(BaseModel):
    # paths
    faces: str = faces_path
    users: str = users_path
    check_users: str = users_check_path
