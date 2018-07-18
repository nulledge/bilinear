from enum import Enum


class Task(Enum):
    Train = 'train'
    Valid = 'valid'

    def to_str(self):
        return str(self)

    def __str__(self):
        return self.value

    @staticmethod
    def from_str(string):
        for task in Task:
            if str(task) == string:
                return task
        return None


tasks = [
    Task.Train,
    Task.Valid,
]
