from enum import Enum


class Category(Enum):
    EDUCATION = 'education', 4
    ENTERTAINMENT = 'entertainment', 2
    HEALTHCARE = 'healthcare', 4
    SUSTENANCE = 'sustenance', 2
    TRANSPORTATION = 'transportation', 1
    TRANSPORTATION2 = 'transportation2', 1

    @property
    def name(self):
        return self.value[0]

    @property
    def distance(self):
        return self.value[1]