# EKC GitHub Organizer -------------------------------------------------

# Versions -------------------------------------------------------------
# 01 - Dec 05th, 2022 - starter.
# 02 - Dec 06th, 2022 - adjusting parameters and added a timer in the
#                       end of the program.
# 03 - Dec 10th, 2022 - Adjusting
# 04 - Dec 30th, 2022 - Adding .txt function
# 05 - 


# Upgrades
# Add an option to consider folders, or a list of folders
# Read an external .json/.txt/.csv with folders data - v04 [Dec 30th, 2022]
#


# Libraries
import os
import shutil
from time import sleep


# Functions
def read_txt(filename, verbose=False):
    """
    Read content from a .txt file and returns as a list.

    """
    file = open(filename, mode="r")
    buffer = file.readlines()
    file.close()

    for i in range(0, len(buffer)):
        buffer[i] = buffer[i].replace("\n", "")

    steps = len(buffer) // 5

    if(verbose == True):
        print(f" > Number of lines in the file: {len(buffer)}")

    return buffer, steps


def remove_tag(string):
    """
    Removes the tag separated by ":"

    """
    string = string.split(":")
    if(len(string) == 2):
        string = string[1].strip()

    else:
        string = None

    return string
    

def remove_folders(file_list):
    """
    Receives a list with folders and files from os.listdir() and
    returns a list with only files (remove folder(s)).
    
    """
    new_list = []
    for file in file_list:
        if(os.path.isfile(file) == True):
            new_list.append(file)

    return new_list


def remove_files(folder_list):
    """
    Receives a list with folders and files from os.listdir() and
    returns a list with only folders (remove file(s)).

    """
    new_list = []
    for folder in folder_list:
        if(os.path.isdir(folder) == True):
            new_list.append(folder)

    return new_list


def transfer_files(file_list, enable_types):
    """
    Receives a list with files from project source and a list with files
    extensions allowed to copy for github destiny.
    Returns a list with files to copy/compare between project source and
    github destiny.

    enable_types is a string with the extensions, need to handle it.

    """
    # Select file extensions to keep (enable_types)
    types_to_handle = enable_types.split(",")

    enable_types = []
    for i in types_to_handle:
        ext = i.strip()
        ext = ext.replace(".", "")
        
        enable_types.append(ext)

    # Select files with extensions enabled
    new_list = []
    for file in file_list:
        name, extension = file.split(".")
        if(enable_types.count(extension) == 1):
            new_list.append(file)

    return new_list


def time_delay(time, verbose=True):
    """
    Delay with a . (dot) indicator.
    
    """
    if(verbose == True):
        print(" ", end="")

    i = 0
    while(i <= time):
        if(verbose == True):
            print(".", end="")

        sleep(1)
        i = i+1

    if(verbose == True):
        print(".")

    return None


# Main Program ---------------------------------------------------------
print("\n ****  Auto Github | Sync project and github folders  **** \n\n") 

# Folders to sync (External info from .txt.
#     name: Name of the folder, suggestion: Folder path,
#    types: String with the types (extensions) that will be sync,
#     root: Folder from Project (Source),
#   github: Folder from github (Destiny),

# External Information -------------------------------------------------
filename = "paths_for_github.txt"
buffer, steps = read_txt(filename)

name, types, root, github = "", "", "", ""

# Sync routine ---------------------------------------------------------

for i in range(0, steps):
    data_list = []
    for j in range(0, 4):
        data = buffer[i*5 + j]
        data = data.split(": ")[1]
        data = data.strip()

        data_list.append(data)

    name, types, root, github = data_list

    # Starting Folder analysis and exchange (if need)
    print(f' > Analyzing {name}')
    update = False
    
    # Files allowed to copy from Project folder
    os.chdir(root)
    project_list = os.listdir()

    project_list = remove_folders(project_list)
    project_list = transfer_files(project_list, types)

    # Files at GitHub folder
    os.chdir(github)
    github_list = os.listdir()

    github_list = remove_folders(github_list)
    github_list = transfer_files(github_list, types)


    # Copying from Project to GitHub -----------------------------------
    for file in project_list:
        _name, extension = file.split(".")
        _name = _name.split("_")
        if(len(_name) > 1):
            name = "_".join(_name[0:-1])
            #version = _name[-1]
            
        else:
            name = _name[0]
            #version = "v00"

        file_noversion = name + "." + extension
        # **** Not using version to compare files (yet) but keeping
        # code for next steps ****

        if(github_list.count(file_noversion) == 0):
            # File does not exists in GitHub = Add file
            update = True
            source = os.path.join(root, file)
            destiny = os.path.join(github, file_noversion)
            shutil.copyfile(source, destiny)
            print(f" >>> New file at github: '{file}'")

        else:
            # File exists in Githhub.
            # Check modification datetime to move the file
            os.chdir(root)
            project_epoch = int(os.path.getmtime(file))

            os.chdir(github)
            github_epoch = int(os.path.getmtime(file_noversion))

            if(project_epoch > github_epoch):
                # Project file is newer than github file = Update
                update = True
                source = os.path.join(root, file)
                destiny = os.path.join(github, file_noversion)
                shutil.copyfile(source, destiny)
                print(f" >>> Updated file at github: '{file}'")


    if(update == True):
        print("")


# Delay of `time` seconds to be able to read all actions done
time_delay(3, verbose=True)


# Cleaning Project Folder ----------------------------------------------
project_folder = r"D:\01 - Projects Binder"
os.chdir(project_folder)

temp_folder = ["__pycache__", ".ipnb_checkpoints"]

folder_list = os.listdir()
folder_list = remove_files(folder_list)

for folder in folder_list:
    new_path = os.path.join(project_folder, folder)
    os.chdir(new_path)

    internal_folder = os.listdir()
    internal_folder = remove_files(internal_folder)

    for temp in temp_folder:
        if(internal_folder.count(temp) == 1):
            folder_remove = os.path.join(new_path, temp)
            shutil.rmtree(folder_remove)
            text = f" > Removing {folder_remove}"
            print(text[-70:])


# Delay of `time` seconds to be able to read all actions done
time_delay(3, verbose=True)
