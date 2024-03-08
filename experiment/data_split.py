import pandas as pd
import sys
import os
import glob
import subprocess as subp
import time

file_path = ""
total_num_entries = 0
num_slices = 0
labels = ""
SPECS_OK = False
file_len = 0

# For path of directories always include "/" at the end
splitted_directory = "./splitted/"
slurm_output_directory = "./slurm_output/"
result_directory = "./results/"

slurm_file = "./casp.slurm"
slurm_partition = "amd-short"
aiz_config = "./azf_config.yml"
clean = True
current_directory = os.getcwd()
experiment_name = "xgboost"


def try_read(file_name):
    global file_path, labels, file_len
    if "./" not in file_name:
        file_name = "./" + file_name

    print("File name: ", file_name)
    try:
        with open(file_name) as f:
            labels = f.readline().strip('\n')
            print("Labels: ", [i for i in labels.split(',')])
            file_path = file_name
            file_len = len(f.readlines())
            print("Length of file: ", file_len)
    except FileNotFoundError:
        new_file_name = input("File not found. Please retype the file path: ")
        print("New file name: ", new_file_name)
        try_read(file_name=new_file_name)


def check_specs():
    global file_path, total_num_entries, num_slices, labels, SPECS_OK, file_len
    SPECS_OK = False
    if file_path is None or file_path == "":
        file_path = str(input("Please fill in a valid file path: "))
        check_specs()

    if total_num_entries == "" or int(total_num_entries) <= 0:
        tmp = input("Please type the desired total number of entries you want: ")
        if tmp != '' and tmp.isdigit():
            total_num_entries = int(tmp)
        else:
            print("Please type a valid integer.")
        check_specs()

    if num_slices == "" or int(num_slices) <= 0:
        tmp = input("Please type the desired number of slices you want: ")
        if tmp != '' and tmp.isdigit():
            num_slices = int(tmp)
        else:
            print("Please type a valid integer.")
        check_specs()

    if total_num_entries < num_slices:
        total_num_entries = num_slices
        check_specs()

    SPECS_OK = True


def check_or_create(mydir):
    CHECK_FOLDER = os.path.isdir(mydir)
    if not CHECK_FOLDER:
        os.makedirs(mydir)


def clean_folder(path, exception_file):
    to_clean = glob.glob(path + "*")
    for f in to_clean:
        if str(f) != exception_file and not os.isdir(f):
            os.remove(f)
    # if not os.listdir(path):
    #     print(path, "is empty. Removing directory...")
    #     os.rmdir(path)
    # else:
    #     print("Exception file exists: ", exception_file,  ", Keeping directory: ", path)


def split_file(split_file, num_slices):
    global splitted_directory, total_num_entries, file_path, clean
    print("Cleaning the splitted and slurm_output directory...")
    if clean:
        clean_folder(splitted_directory, split_file)
        clean_folder(slurm_output_directory, "")

    file_len = 0
    with open(split_file, 'r') as file:
        content = file.readlines()
        
    # Calculate the number of lines per split
    num_lines = len(content)
    lines_per_split = num_lines // num_slices
    remaining_lines = num_lines % num_slices

    start = 0
    for i in range(num_slices):
        # Calculate the end index for the current split
        end = start + lines_per_split + (1 if i < remaining_lines else 0)

        # Create a new file for the split
        small_file_path = splitted_directory + "smile_split_" + str(i + 1) + ".txt"
        with open(small_file_path, 'w') as small_file:
            small_file.writelines(content[start:end])

        start = end

    print(f"The first {total_num_entries} lines of {file_path} has been separated into '{str(split_file)}' which has been split into {num_slices} smaller files.")


def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num - 1] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()
    

def to_camel_case(text):
    s = text.replace("-", " ").replace("_", " ")
    s = s.split()
    if len(text) == 0:
        return text
    return s[0] + ''.join(i.capitalize() for i in s[1:])


def submit_slurm():
    global slurm_file, num_slices, splitted_directory, slurm_output_directory, slurm_partition, aiz_config
    ans_1 = str(input("Slurm file modified. Start sbatch? (Y/N)\n"))
    if ans_1 == "Y" or ans_1 == "y":
        count = 0
        while count < num_slices:
            time.sleep(1)
            for root_dir, cur_dir, files in os.walk(splitted_directory):
                count = len(files)
            print('Waiting for file split to finish...file count under splitted:', count)
        print("Current squeue: ")
        subp.check_call("squeue", shell=True)

        ans_2 = ""
        while not any(ans_2 == s for s in ['Y', 'y', 'N', 'n']):
            ans_2 = str(input("The current partition is: " + slurm_partition + ". Do you want to change it? (Y/N) \n"))
        if ans_2 == 'Y' or ans_2 == 'y':
            print("\n                 PARTITION         TIMELIMIT  \n \
                testing             1:00:00 \n \
                cpu-short*          4:00:00 \n \
                cpu-medium       1-00:00:00 \n \
                cpu-long         7-00:00:00 \n \
                gpu-short           4:00:00 \n \
                gpu-medium       1-00:00:00 \n \
                gpu-long         7-00:00:00 \n \
                mem              14-00:00:0 \n \
                amd-short           4:00:00")
            allowed_partition = {
                'testing' : "1:00:00", 
                'cpu-short' : "4:00:00", 
                'cpu-medium' : "1-00:00:00", 
                'cpu-long' : "7-00:00:00", 
                'gpu-short' : "4:00:00", 
                'gpu-medium' : "1-00:00:00", 
                'gpu-long' : "7-00:00:00", 
                'mem' : "14-00:00:0", 
                'amd-short' : "4:00:00"
            }
            partition_choice = ""
            while not any(partition_choice == s for s in allowed_partition):
                partition_choice = str(input("Please select a partition:\n"))
            slurm_partition = partition_choice
            print("partition selected as: ", slurm_partition)
            replace_line(slurm_file, 3, "#SBATCH --partition=\"" + slurm_partition + "\"\n")
            replace_line(slurm_file, 4, "#SBATCH --time=" + allowed_partition[slurm_partition] + "\n")
        elif ans_2 == 'N' or ans_2 == 'n':
            print("Default partition is being used: ", slurm_partition)

        ans_3 = ""
        while not any(ans_3 == s for s in ['Y', 'y', 'N', 'n']):
            ans_3 = str(input("The current AIZ config file is: " + aiz_config + ". Do you want to change it? (Y/N) \n"))
        if ans_3 == 'Y' or ans_3 == 'y':
            conf_choice = ""
            while not os.path.isfile(conf_choice):
                conf_choice = str(input("Please select a AIZ config file:\n"))
            aiz_config = conf_choice
            print("AIZ config file selected as: ", aiz_config)
            replace_line(slurm_file, 17, "config=\"" + current_directory + "/" + aiz_config.split("/")[-1] + "\"" + "\n")
        elif ans_3 == 'N' or ans_3 == 'n':
            print("Default AIZ config file is being used: ", aiz_config)

        # subp.check_call("mv " + slurm_file + " " + slurm_output_directory, shell=True)
        # subp.check_call("sbatch " + slurm_output_directory + slurm_file.split("/")[-1], shell=True)
        # subp.check_call("squeue --me", shell=True)
        # subp.check_call("mv " + slurm_output_directory + slurm_file.split("/")[-1] + " ./", shell=True)

        subp.check_call("sbatch " + slurm_file, shell=True)
        subp.check_call("squeue --me", shell=True)
    elif ans_1 == "N" or ans_1 == "n":
        print("Process finished. Run sbatch manually to submit jobs.")
    else:
        submit_slurm()



if len(sys.argv) < 4:
    # print(sys.argv)
    print("Usage: python data_split.py [file_name] [total_number_of_entries] [number_of_slices]")
    check_specs()
else:
    try:
        file_path = str(sys.argv[1])
        total_num_entries = int(sys.argv[2])
        num_slices = int(sys.argv[3])
        check_specs()
        print("File Path: ", file_path, "  |  Total number of entries: ", total_num_entries, "  |  Number of Slices: ", num_slices)
    except:
        print("Not enough arguments...")
        check_specs()

# ================================================================== PROGRAM STARTS HERE ===================================================================
if SPECS_OK:
    if num_slices > 1001:
        print("---------------------------------------------------WARNING-------------------------------------------------------\n \
              Slice number greater than 1001 will not be accepted by sbatch. \n \
            -----------------------------------------------------------------------------------------------------------------\n")
    try_read(file_path)
    labels_list = labels.split(',')
    print("Directory of splitted files: ", splitted_directory)
    print("Slurm file: ", slurm_file)

    exp_ans = ""
    while exp_ans == "" or not exp_ans.isalnum():
        exp_ans = input("Please type in the name of the experiment: (Letters and digits only) \n")
    experiment_name = to_camel_case(exp_ans)
    print("Experiment name selected: ", experiment_name)

    check_or_create(result_directory + experiment_name + "/")
    check_or_create(slurm_output_directory + experiment_name + "/")
    check_or_create(splitted_directory + experiment_name + "/")

    result_directory = result_directory + experiment_name + "/"
    slurm_output_directory = slurm_output_directory + experiment_name + "/"
    splitted_directory = splitted_directory + experiment_name + "/"

    df = pd.read_csv(file_path)
    df.columns = labels_list
    df = df.head(n=total_num_entries)
    output_file = splitted_directory + "smile_" + str(total_num_entries) + ".txt"
    df['clean_smiles'].to_csv(output_file, header=None, index=None, mode='w')

    while not os.path.isfile(output_file):
        print("waiting for splitted file to be generated...")

    split_file(output_file, num_slices)
    # line number starts with 1
    slurm_array_size = "#SBATCH --array=1-" + str(num_slices) + "\n"
    slurm_smiles = "smiles_csv=\"" + current_directory + splitted_directory.split(".")[-1] + "smile_split_${ROW_INDEX}.txt\"" + "\n"
    slurm_out_script = "#SBATCH --output=" + current_directory + slurm_output_directory.split(".")[-1] + "slurm-%A_%a.out" + "\n"
    slurm_result_output = "output=\"" + current_directory + "/results/" + experiment_name + "/" + experiment_name + "_${ROW_INDEX}_result.hdf\"" + "\n"
    slurm_conf = "config=\"" + current_directory + "/" + aiz_config.split("/")[-1] + "\"" + "\n" 
    replace_line(slurm_file, 9, slurm_array_size)
    replace_line(slurm_file, 10, slurm_out_script)
    replace_line(slurm_file, 16, slurm_smiles)
    replace_line(slurm_file, 17, slurm_conf)
    replace_line(slurm_file, 18, slurm_result_output)
    
    # replace_line(aiz_config, 21, "" + "\n")
    # replace_line(aiz_config, 22, "" + "\n")
    # replace_line(aiz_config, 25, "" + "\n")
    # replace_line(aiz_config, 28, "" + "\n")

    submit_slurm()
else:
    check_specs()
