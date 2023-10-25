import os 


# script that runs all the experiments presented in the paper
function_to_run = "run_main_config.py"
config_folder_list = ["table_1", "table_2", "table_3", "table_4"]
run_name_list = ["table_1", "table_2", "table_3", "table_4"]
tasks_list = ["mnist", "fmnist", "cifar10", "cifar100"]

N = len(tasks_list)

for k in range(len(config_folder_list)):
   config_folder = config_folder_list[k]
   run_name = run_name_list[k]

   for n in range(N):

      data = tasks_list[n]
      if config_folder=="table_1" or config_folder=="table_2":
         config_names = ["pepita_" + data, "bp_" + data]
      elif config_folder=="table_3" or config_folder=="table_4":
         config_names = ["pepita_" + data + "_adv", "bp_" + data+ "_adv"]
      
      M = len(config_names)
      
      for m in range(M):

         config_name = config_names[m]
         path_to_config = "configs/" + config_folder + "/" + config_name + ".yaml"

         # if you have cuda, add "CUDA_VISIBLE_DEVICES=number_of_device" before "python3"
         os.system("python3 " + function_to_run + " -path_to_config " + path_to_config + " -run_name " + run_name)