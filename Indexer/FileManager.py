import os


class FileManager:
    def __init__(self):
        pass

    @staticmethod
    def get_all_file_names(dire):
        file_paths = list()
        for path, subdirs, files in os.walk(dire):
            for name in files:
                file_paths.append(os.path.join(path, name))
        return file_paths
