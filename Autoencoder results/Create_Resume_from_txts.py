import os
import re


def prepend_line(file_name, line: str):
    """ Insert given string as a new line at the beginning of a file """
    # define name of temporary dummy file
    dummy_file = file_name + '.bak'
    # open original file in read mode and dummy file in write mode
    with open(file_name, 'r') as read_obj, open(dummy_file, 'w') as write_obj:
        # Write given line to the dummy file
        write_obj.write(line + '\n')
        # Read lines from original file one by one and append them to the dummy file
        for line in read_obj:
            write_obj.write(line)
    # remove original file
    os.remove(file_name)
    # Rename dummy file as the original file
    os.rename(dummy_file, file_name)


def select_sub_folder(folder: str = "."):
    for (root, dirs, files) in os.walk(folder):
        pass

rex = re.compile("^Autoencoder_(S_(\d+_)+|E_(\d+_)D_(\d+_)*)(\S+)\.txt$")
list_dir = os.listdir(".")
subdir = [i for i in list_dir if os.path.isdir(i) and i.startswith("Autoencoder_")]
resume = [i for i in list_dir if i.startswith("Resume")]
for to_reset in resume:
    os.remove(to_reset)
for root, dirs, files in os.walk("."):
    for name in files:
        match = rex.match(name)
        if match is not None:
            defect_str = match.group(5)
            with open("_".join(["Resume", defect_str]) + ".txt", "a") as f:
                f.write(name)
                f.write(":\n")
                h_name = open(os.path.join(root, name))
                f.writelines(h_name.readlines())