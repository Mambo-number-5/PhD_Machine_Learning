import os
import re

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