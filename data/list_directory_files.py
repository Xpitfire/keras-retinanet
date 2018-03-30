import glob
import sys
import os

if __name__ == "__main__":
    dir_path = sys.argv[1]
    out_file_name = sys.argv[2]

    out_list = []
    for file in glob.iglob(os.path.join(dir_path, '*')):
        if file.endswith('.jpg') or file.endswith('.png'):
            out_list.append(file)

    with open(out_file_name, 'w') as fdata:
        for item in out_list:
            fdata.write("{}\n".format(item))
