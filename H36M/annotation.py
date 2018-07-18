from enum import Enum
from .task import Task


class Annotation(Enum):
    S = 'S'  # 3D position
    Center = 'center'  # 2D position in image space
    Part = 'part'  # 2D position
    Scale = 'scale'  # Bounding box scale
    Z = 'zind'  # Depth index of voxel
    Image = 'image'  # Image name

    def to_str(self):
        return str(self)

    def __str__(self):
        return self.value


annotations = dict()
annotations[str(Task.Train)] = [
    Annotation.S,
    Annotation.Center,
    Annotation.Part,
    Annotation.Scale,
    Annotation.Z,
]
annotations[str(Task.Valid)] = [
    Annotation.S,
    Annotation.Center,
    Annotation.Part,
    Annotation.Scale,
    Annotation.Z,
]
