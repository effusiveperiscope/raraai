import os 

def win_longpath(path):
    if os.name != 'nt':
        return path
    if path.startswith('\\\\?\\'):
        return path
    return '\\\\?\\' + os.path.abspath(path)