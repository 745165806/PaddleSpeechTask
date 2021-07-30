"""Data preparation for punctuation_restoration task."""

from speechtask.punctuation_restoration.utils.default_parser import default_argument_parser
from speechtask.punctuation_restoration.utils.utility import print_arguments
from speechtask.punctuation_restoration.utils.punct_prepro import process_data
import yaml
import os

# create dataset from raw data files
def main(config, args):
    print("Start preparing data from raw data.")

    if not os.path.exists(config["save_path"]) or not os.listdir(config["save_path"]):
        print('111111')
        process_data(config)
    if not os.path.exists(config["pretrained_emb"]) and config["use_pretrained"]:
        print('222222')
        process_data(config)

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



