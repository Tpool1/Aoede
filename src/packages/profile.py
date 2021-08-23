from core import core
import os

class profile(core):

    def __init__(self, name):
        self.name = name

        try:
            self.make_data_path()
        except FileExistsError:
            pass

        # make empty conversation data file for profile
        open(os.path.join(self.data_path, "conversations.txt"), mode="a").close()

        with open(os.path.join(self.data_path, "info.txt"), mode='w') as f:
            f.write("Name: " + self.name)

    def make_data_path(self):
        root = "data\\profiles"
        self.data_path = os.path.join(root, self.name)

        os.mkdir(self.data_path)
