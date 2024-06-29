import os
import subprocess as subp
import time

experiment_name = ""
start_index = "0"
end_index = "0"

current_directory = os.getcwd()
splitted_directory = "./splitted/"
slurm_output_directory = "./slurm_output/"
result_directory = "./results/"
aiz_config = "./azf_config_200k.yml"
slurm_file = "./casp.slurm"
slurm_partition = "amd-short"


def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num - 1] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

def submit_slurm():
    global slurm_file, splitted_directory, slurm_output_directory, slurm_partition, aiz_config
    ans_1 = str(input("Slurm file modified. Start sbatch? (Y/N)\n"))
    if ans_1 == "Y" or ans_1 == "y":
        # count = 0
        # while count < num_slices:
        #     time.sleep(1)
        #     for root_dir, cur_dir, files in os.walk(splitted_directory):
        #         count = len(files)
        #     print('Waiting for file split to finish...file count under splitted:', count)
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
                amd-short           4:00:00 \n \
                amd-long         7-00:00:00 \n \
                amd-gpu-short       4:00:00 \n \
                amd-gpu-long     7-00:00:00 \n \
                  ")
            allowed_partition = {
                'testing' : "1:00:00", 
                'cpu-short' : "4:00:00", 
                'cpu-medium' : "1-00:00:00", 
                'cpu-long' : "7-00:00:00", 
                'gpu-short' : "4:00:00", 
                'gpu-medium' : "1-00:00:00", 
                'gpu-long' : "7-00:00:00", 
                'mem' : "14-00:00:0", 
                'amd-short' : "4:00:00",
                'amd-long' : "7-00:00:00", 
                'amd-gpu-short' : "4:00:00",
                'amd-gpu-long' : "7-00:00:00",
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


        subp.check_call("sbatch " + slurm_file, shell=True)
        subp.check_call("squeue --me", shell=True)
    elif ans_1 == "N" or ans_1 == "n":
        print("Process finished. Run sbatch manually to submit jobs.")
    else:
        submit_slurm()

while experiment_name == "":
    experiment_name = input("Enter the experiment name: ")
print("Experiment name selected: ", experiment_name)

result_directory = result_directory + experiment_name + "/"
slurm_output_directory = slurm_output_directory + experiment_name + "/"
splitted_directory = splitted_directory + experiment_name + "/"

while int(end_index) <= int(start_index):
    end_index = input("Enter the end index: ")
    while not end_index.isdigit():
        end_index = input("Enter the end index: ")
    start_index = input("Enter the start index: ")
    while not start_index.isdigit():
        start_index = input("Enter the start index: ")

start_index = int(start_index)
end_index = int(end_index)
print("Start index: ", start_index, ", End index: ", end_index)

results_dir = "./results/" + experiment_name + "/"
file_names = os.listdir(results_dir)
existing_indexes = [int(f.split("_")[-2]) for f in file_names]
indexes = list(range(start_index, end_index + 1))
missing_indices = [i for i in indexes if i not in existing_indexes]
missing_indices_str = str(missing_indices).replace(" ", "").replace("[", "").replace("]", "")

missing_indices.sort()

missing_indices_interval = []
interval_begin = missing_indices[0]
interval_end = missing_indices[0]

for i in range(1, len(missing_indices)):
    if missing_indices[i] == interval_end + 1:
        interval_end = missing_indices[i]
    else:
        if interval_begin == interval_end:
            missing_indices_interval.append(str(interval_begin))
        else:
            missing_indices_interval.append(str(interval_begin) + "-" + str(interval_end))
        interval_begin = missing_indices[i]
        interval_end = missing_indices[i]

if interval_begin == interval_end:
    missing_indices_interval.append(str(interval_begin))
else:
    missing_indices_interval.append(str(interval_begin) + "-" + str(interval_end))

missing_indices_interval_str = ",".join(missing_indices_interval)
print("Missing indices:", missing_indices_str)
print("Missing indices interval:", missing_indices_interval_str)

slurm_array_size = "#SBATCH --array=" + missing_indices_interval_str + "\n"
slurm_smiles = "smiles_csv=\"" + current_directory + splitted_directory.split(".")[-1] + "smile_split_${ROW_INDEX}.txt\"" + "\n"
slurm_out_script = "#SBATCH --output=" + current_directory + slurm_output_directory.split(".")[-1] + "slurm-%A_%a.out" + "\n"
slurm_result_output = "output=\"" + current_directory + "/results/" + experiment_name + "/" + experiment_name + "_${ROW_INDEX}_result.hdf\"" + "\n"
slurm_conf = "config=\"" + current_directory + "/" + aiz_config.split("/")[-1] + "\"" + "\n" 
slurm_memory = str(input("Enter the integer amount of memory (e.g. 50): "))
slurm_memory = "#SBATCH --mem=" + slurm_memory + "GB" + "\n"

replace_line(slurm_file, 6, slurm_memory)
replace_line(slurm_file, 9, slurm_array_size)
replace_line(slurm_file, 10, slurm_out_script)
replace_line(slurm_file, 16, slurm_smiles)
replace_line(slurm_file, 17, slurm_conf)
replace_line(slurm_file, 18, slurm_result_output)


submit_slurm()
