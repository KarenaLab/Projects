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
#
# 10 - Mar 21th, 2023 - Adding two destinies (Zen-14 and Book2)
# 11 - 


# Upgrades
# Read an external .json/.txt/.csv with folders data - v04 [Dec 30th, 2022]
# Automatic remove of files at github that are not at folder
# Add an option to consider folders, or a list of folders (bigger projects)
# 


# Libraries
import os
import shutil

import socket
from time import sleep


# Setup/Config
time_froozen = 3
path_script = os.getcwd()
path_modules = r"c:\python_modules"
path_projects = r"D:\01 - Projects Binder"


# Functions

def pc_choose():
    """
    Read PC and returns the folder to append to GitHub folder path.

    """
    pc_name = socket.gethostname()

    if(pc_name == "EKC-Zen14"):
        github_prefix = r"D:\02a - GIT EKC-Zen14"

    elif(pc_name == "EKC-Book2"):
        github_prefix = r"D:\02b - GIT EKC-Book2"

    else:
        github_prefix = None
        print(f" *** Error: PC not registered for github sync {pc_name} ***")


    return github_prefix
    

def read_txt(filename, lines=5, verbose=False):
    """
    Extracts information from a .txt.
    Returns information as a list and the number of steps to process.

    """
    # Read information from .txt
    file = open(filename, mode="r")
    buffer = file.readlines()
    file.close()

    # Prepare information as a list
    for i in range(0, len(buffer)):
        buffer[i] = buffer[i].replace("\n", "")

    blocks = len(buffer) // (lines + 1)

    # Group information
    info_list = []
    for i in range(0, blocks):
        info_dict = {}
        for j in range(0, lines+1):
            data = buffer.pop(0)
            if(data != ""):
                field, value = data.split(": ")
                info_dict[field] = value

        info_list.append(info_dict)


    if(verbose == True):
        print(f" > Nuber of folders to compare and transfer: {steps}")


    return info_list    
    

def files_list(path=None):
    """
    Returns a list with only file(s) (remove folder(s)) from given path
    or the current path.
    
    """
    if(path != None):
        path_comeback = os.getcwd()
        os.chdir(path)

    content = os.listdir()
    
    files_list = []
    for f in content:
        if(os.path.isfile(f) == True):
            files_list.append(f)

    if(path != None):
        os.chdir(path_comeback)
        

    return files_list


def folders_list(path=None):
    """
    Returns a list with only folder(s) (remove file(s)) from given path
    or the current path.
    
    """
    if(path != None):
        path_comeback = os.getcwd()
        os.chdir(path)

    content = os.listdir()
    
    folders_list = []
    for f in content:
        if(os.path.isdir(f) == True):
            folders_list.append(f)

    if(path != None):
        os.chdir(path_comeback)
        

    return folders_list


def transfer_files(file_list, enable_types):
    """
    Receives a **list with file(s)** from project source
    and a **list with extension(s)** allowed to copy for github destiny.
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
    for f in file_list:
        name, extension = f.split(".")
        if(enable_types.count(extension) == 1):
            new_list.append(f)


    return new_list


def prepare_module_files(text):
    """
    Receives a string from paths_to_sync.txt and transforms into data
    to be processed.

    """
    if(text == "None"):
        module_files = []

    else:
        module_files = []

        text = text.split(",")       
        for i in text:
            new_file = i.strip()
            module_files.append(new_file)


    return module_files


def remove_index(filename):
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


def inner_join(left, right):
    """
    Performs **inner join** between **left** list and **right** list.
    """
    i_join = []
    for i in right:
        if(left.count(i) > 0):
            i_join.append(i)

    return i_join


def left_join(left, right):
    """
    Performs **left join** between **left** list and **right** list.
    Attention: Take care with left and right position.
    """
    l_join = []
    for i in left:
        if(right.count(i) == 0):
            l_join.append(i)

    return l_join


def outter_join(left, right):
    """
    Performs **outter join** between **left** list and **right** list.
    """
    join = left + right
    o_join = []

    for i in join:
        if(o_join.count(i) == 0):
            o_join.append(i)

    return o_join


def remove_temp_folders(path):
    """
    Removes python (__pychache__) and Jupyter (.ipnb_checkpoints)
    temporary folders

    """
    os.chdir(path)
    temp_folder = ["__pycache__", ".ipnb_checkpoints"]

    projects_list = folders_list()
    for folder in projects_list:
        os.chdir(os.path.join(path, folder))
        internal_folders = folders_list()

        for temp in temp_folder:
            if(internal_folders.count(temp) == 1):
                folder_remove = os.path.join(os.getcwd(), temp)
                shutil.rmtree(folder_remove)
                text = f"> Removing {folder_remove}"
                print_info(text)


    return None


def print_info(text, cut=12, connector="... ", limit=70):
    """


    """
    if(len(text) > limit):
        leftover = (-1) * (limit - cut - len(connector))
        text = text[0:cut] + "... " + text[leftover: ]

    print(text)
    
    return None


def time_delay(time, verbose=True):
    """
    Delay with a . (dot) indicator.
    
    """
    if(verbose == True):
        print("  ", end="")

    for i in range(0, time):
        if(verbose == True):
            print(".", end="")

        sleep(1)
        
    print("")
    
    return None


# Main program ---------------------------------------------------------
print("\n ****  Auto Github 2 | Sync project and github* folders  **** \n") 

# Folders to sync (External info from .txt.
#   name: Name of the folder, suggestion: Folder path,
#  types: String with the types (extensions) that will be sync,
#   root: Folder from Project (Source),
# github: Folder from github (Destiny),
# module: File(s) to be sync with C:\python_modules


# External information
filename = "paths_for_sync.txt"
buffer = read_txt(filename)
github_prefix = pc_choose()


# Getting information from modules folder
module_lake = files_list(path_modules)

#module_source = transfer_files(module_source, ".py")

for i in range(0, len(buffer)):
    data = buffer[i]

    print(f'> Folder: {data["name"]}')
    update = False
    
    types = data["types"]
    path_root = data["root"]
    path_github = os.path.join(github_prefix, data["github"])
    
    os.chdir(path_root)
    root_files = transfer_files(files_list(), types)
    github_files = transfer_files(files_list(path_github), types)
    module_files = prepare_module_files(data["module"])


    # Project for GitHub    
    for filename in root_files:
        filename_noindex = remove_index(filename)

        if(github_files.count(filename_noindex) == 0):
            # File does not exists in github = Add file          
            update = True
            source = os.path.join(path_root, filename)
            destiny = os.path.join(path_github, filename_noindex)
            shutil.copyfile(source, destiny)
            print_info(f' >>> New file at github: "{filename_noindex}"')

        else:
            # File exists in github = Check if need to update
            os.chdir(path_root)
            project_epoch = int(os.path.getmtime(filename))

            os.chdir(path_github)
            github_epoch = int(os.path.getmtime(filename_noindex))

            if(project_epoch > github_epoch):
                update = True
                source = os.path.join(path_root, filename)
                destiny = os.path.join(path_github, filename_noindex)
                shutil.copyfile(source, destiny)
                print_info(f' >>> Updated file at github: "{filename_noindex}"')


        if(module_files.count(filename_noindex) == 1):
            # File to be copied to modules lake

            if(module_lake.count(filename_noindex) == 0):
                # File does not exists in modules lake = Add file          
                update = True
                source = os.path.join(path_root, filename)
                destiny = os.path.join(path_modules, filename_noindex)
                shutil.copyfile(source, destiny)
                print_info(f' >>> New file at modules lake: "{filename_noindex}"')

            else:
                # File exists in modules lake = Check if need to update
                os.chdir(path_root)
                project_epoch = int(os.path.getmtime(filename))

                os.chdir(path_modules)
                github_epoch = int(os.path.getmtime(filename_noindex))

                if(project_epoch > github_epoch):
                    update = True
                    source = os.path.join(path_root, filename)
                    destiny = os.path.join(path_modules, filename_noindex)
                    shutil.copyfile(source, destiny)
                    print_info(f' >>> Updated file at modules: "{filename_noindex}"')

               
    if(update == True):
        print("")
        

time_delay(time_froozen)


# Cleaning Project Folder ----------------------------------------------
remove_temp_folders(path_projects)
time_delay(time_froozen)


# end
