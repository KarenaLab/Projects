# EKC GitHub Organizer -------------------------------------------------

# Versions -------------------------------------------------------------
# 01 - Dec 05th, 2022 - starter.
# 02 - Dec 06th, 2022 - adjusting parameters and added a timer in the
#                       end of the program.
# 03 - Dec 10th, 2022 - Adjusting
# 04 - Dec 30th, 2022 - Adding .txt function
# 05 - Jan 10th, 2023 - Adjusting name extension
# 06 - Jan 28th, 2023 - Adding a module option
# 07 - Feb 10th, 2023 - Sync github and module update
# 08 - Feb 27th, 2023 - Adding two destinies (Zen-14 and Book2)
# 09 - 


# Upgrades
# Read an external .json/.txt/.csv with folders data - v04 [Dec 30th, 2022]
# Automatic remove of files at github that are not at folder
# Add an option to consider folders, or a list of folders (bigger projects)
# 


# Libraries
import os
import shutil
from time import sleep


# Setup/Config
time = 3


# Functions ------------------------------------------------------------
def read_txt(filename, verbose=False):
    """
    Read content from a .txt file and returns as a list.

    """
    file = open(filename, mode="r")
    buffer = file.readlines()
    file.close()

    for i in range(0, len(buffer)):
        buffer[i] = buffer[i].replace("\n", "")

    steps = len(buffer) // 6

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


def filename_treat(filename):
    """
    file "_vxx" extension analysis and decison to remove for github
    folder, not preserving version control but keeping it always the
    last version.
    
    """
    _name, extension = filename.split(".")
    _name = _name.split("_")

    if(len(_name) > 1):
        version = _name[-1]

        if(version[0] == "v" and version[1: ].isdigit() == True):
            name = "_".join(_name[0:-1])
            version = _name[-1]

        else:
            name = "_".join(_name)
            version = ""
                
    else:
        name = _name[0]
        version = ""

    filename = name + "." + extension

    return filename


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
print("\n ****  Auto Github | Sync project and github folders  **** \n") 

# Folders to sync (External info from .txt.
#   name: Name of the folder, suggestion: Folder path,
#  types: String with the types (extensions) that will be sync,
#   root: Folder from Project (Source),
# github: Folder from github (Destiny),
# module: File(s) to be sync with C:\python_modules


# External Information
filename = "paths_for_sync.txt"
buffer, steps = read_txt(filename)

# Getting information from modules folder
module_path = r"c:\python_modules"
os.chdir(module_path)
module_source = os.listdir()
module_source = remove_folders(module_source)
module_source = transfer_files(module_source, ".py")


# Sync routine ---------------------------------------------------------
for i in range(0, steps):
    data_list = []

    # Getting name, types, root and github (sequential **no_lines** rows)
    no_lines = 5
    for j in range(0, no_lines):
        data = buffer[i*(no_lines+1) + j]
        data = data.split(": ")[1]
        data = data.strip()

        data_list.append(data)
        
    name, types, root, github, module = data_list

    # Starting folder analysis and exchange (if need)
    print(f' > Analyzing {name}')
    update = False
    
    # Files allowed to copy from Project folder
    os.chdir(root)
    project_list = os.listdir()

    project_list = remove_folders(project_list)
    project_list = transfer_files(project_list, types)
   
    # Files at **GitHub** folder
    github_list = []    
    if(github != "None"):
        os.chdir(github)
        github_list = os.listdir()

        github_list = remove_folders(github_list)
        github_list = transfer_files(github_list, types)


    # List for **Module** folder
    module_list = []
    if(module != "None"):
        module = module.split(",")
        for i in module:
            filename = i.strip()
            module_list.append(filename)
        

    # Copying from Project to GitHub -----------------------------------
    for file in project_list:
        filename = filename_treat(file)

        # **** Not using version to compare files (yet) but keeping
        # code for next steps ****

        # Project for Github
        if(github_list.count(filename) == 0):
            # File does not exists in GitHub and Module = Add file
            update = True
            source = os.path.join(root, file)
            destiny = os.path.join(github, filename)
            shutil.copyfile(source, destiny)
            print(f" >>> New file at github: '{filename}'")

            if(module_list.count(filename) > 0):
                source = os.path.join(root, file)
                destiny = os.path.join(module_path, filename)
                shutil.copyfile(source, destiny)
                print(f" >>> Updated file at modules: '{filename}'")
            
        else:
            # File exists in Github.
            # Check modification datetime to move the file
            os.chdir(root)
            project_epoch = int(os.path.getmtime(file))

            os.chdir(github)
            github_epoch = int(os.path.getmtime(filename))

            if(project_epoch > github_epoch):
                # Project file is newer than github file = Update
                update = True
                source = os.path.join(root, file)
                destiny = os.path.join(github, filename)
                shutil.copyfile(source, destiny)
                print(f" >>> Updated file at github: '{filename}'")

                if(module_list.count(filename) > 0):
                    # Updating python modules
                    destiny = os.path.join(module_path, filename)
                    shutil.copyfile(source, destiny)
                    print(f" >>> Updated file at modules: '{filename}'")


    if(update == True):
        print("")

# Delay of `time` seconds to be able to read all actions done
# variable `time` at setup/config
time_delay(time, verbose=False)


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

            # Adjusting to fit at half page (max=70)
            if(len(text) > 70):
                text = text[0:12] + "... " + text[-54: ]
                
            print(text)


# Delay of `time` seconds to be able to read all actions done
# variable `time` at setup/config
time_delay(time, verbose=False)


# end
