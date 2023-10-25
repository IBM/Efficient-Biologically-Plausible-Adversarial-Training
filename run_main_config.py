import main_config

def run_config(path_to_config, run_name):
   main_config.main(path_to_config, run_name)

if __name__ == '__main__':
   import argparse
   parser = argparse.ArgumentParser()
   parser.add_argument('-path_to_config', type=str, default='/u/matyf/code/bio-plausible-models/neurips/configs/final/pepita_mnist.yaml', help='Path to config to run.')
   parser.add_argument('-run_name', type=str, default='test', help='Name of folder where config is.')
   args = parser.parse_args()
   run_config(args.path_to_config, args.run_name)