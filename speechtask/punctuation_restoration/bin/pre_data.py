"""Data preparation for punctuation_restoration task."""

from speechtask.punctuation_restoration.utils.default_parser import default_argument_parser
from speechtask.punctuation_restoration.utils.utility import print_arguments
from speechtask.punctuation_restoration.utils.punct_pre import process_chinese
import yaml
import os

# create dataset from raw data files
def main(config, args):
    print("Start preparing data from raw data.")
    if(config["language"]=="chinese"):
        process_chinese(config)
    elif(config["language"]=="english"):
        process_chinese(config)


    print("Finish preparing data.")

    

if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print_arguments(args, globals())

    # https://yaml.org/type/float.html
    with open(args.config,"r") as f:
        config= yaml.load(f,Loader=yaml.FullLoader)

    # config.freeze()
    print(config)
    main(config["data"], args)



