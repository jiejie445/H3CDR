
import  os
"""
:param k: 当前路径后退级数
:return: 后退k级后的目录
"""
def dir_path(k=1):
    fpath = os.path.realpath(file)
    dir_name = os.path.dirname(fpath)
    dir_name = dir_name.replace("\\", "/")
    p = len(dir_name) - 1
    while p > 0:
        if dir_name[p] == "/":
            k -= 1
        if k == 0:
            break
    p -= 1
    p += 1
    dir_name = dir_name[0: p]
    return dir_name