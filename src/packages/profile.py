from core import core
import os

class profile(core):

    def __init__(self, name):
        self.name = name

        try:
            self.make_data_path()
        except FileExistsError:
            pass

        self.user_data = os.listdir(self.data_path)

    def make_data_path(self):
        root = "data\\profiles"
        self.data_path = os.path.join(root, self.name)

        os.mkdir(self.data_path)

        os.mkdir(os.path.join(self.data_path, "conversations.txt"))
