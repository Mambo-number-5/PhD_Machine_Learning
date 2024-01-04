import os
from os.path import join


def create_resume_pcas(folder: str = None, defects: [str] = None) -> None:
    """
    Create a resume (concatenation of text files) for a folder which has sub-folders with name that begin with "PCA_"
    which themselves have files with a name *_(defect).txt (where defect is a string contained in the
    list parameter: defects) to concatenate
    :param folder: folder in which search the sub-folders, if None the os.listdir function
    is called on "." folder (as the os documentation states)
    :param defects: List of name of defects that are used in the file as *_(defect).txt, if None the default
    value is ["DEPRESS", "CRICCA", "RIGA"]
    :return: None
    """
    if defects is None:
        defects = ["DEPRESS", "CRICCA", "RIGA", "ANOMALY"]
    files_handle = dict()

    for defect in defects:
        name_file = "Resume_PCA_" + defect + ".txt"
        files_handle[defect] = open(join(folder, name_file), "w")
    dict_pca = dict()
    for i in os.listdir(folder):
        if os.path.isdir(join(folder, i)) and i.startswith("PCA_"):
            dict_pca[int(i[4:])] = i
    for pca in sorted(dict_pca.keys(), reverse=True):
        for file in os.listdir(join(folder, dict_pca[pca])):
            if file.endswith(".txt"):
                for defect in defects:
                    if defect in file:
                        with open(join(folder, dict_pca[pca], file)) as f:
                            for line in f:
                                files_handle[defect].write(line)
    for v in files_handle.values():
        v.close()


if __name__ == "__main__":
    current_folder = "."
    autoencoder_folders = [i for i in os.listdir(current_folder) if os.path.isdir(join(current_folder, i))
                           and i.startswith("Autoencoder_")]
    for i in autoencoder_folders:
        create_resume_pcas(folder=i)