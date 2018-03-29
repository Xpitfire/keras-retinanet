import random
import sys
import codecs

if __name__ == "__main__":
    in_file_name = sys.argv[1]
    out_file_name = sys.argv[2]
    num_of_lines = sys.argv[3]

    out_list = []
    with codecs.open(in_file_name, "r", encoding='utf-8', errors='ignore') as fdata:
        lines = fdata.read().splitlines()
        for _ in range(int(num_of_lines)):
            line = random.choice(lines)
            entries = line.split("\t")
            print(entries)
            out_list.append(entries[1])

    with open(out_file_name, 'w') as fdata:
        for item in out_list:
            fdata.write("{}\n".format(item))
