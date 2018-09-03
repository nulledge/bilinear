from .task import Task


class Annotation:
    S = 'S'  # 3D position
    Center = 'center'  # 2D position in image space
    Part = 'part'  # 2D position
    Scale = 'scale'  # Bounding box scale
    Image = 'image'  # Image name
    Root_Of = 'root of '  # Root position
    Mean_Of = 'mean of '  # Mean of position
    Stddev_Of = 'stddev of '  # Stddev of position


annotations = dict()
annotations[Task.Train] = [
    Annotation.S,
    Annotation.Center,
    Annotation.Part,
    Annotation.Scale,
]
annotations[Task.Valid] = [
    Annotation.S,
    Annotation.Center,
    Annotation.Part,
    Annotation.Scale,
]
