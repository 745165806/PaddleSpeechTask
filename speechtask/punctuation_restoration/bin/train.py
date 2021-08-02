# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trainer for punctuation_restoration task."""
from paddle import distributed as dist

from speechtask.punctuation_restoration.training.trainer import Trainer
from speechtask.punctuation_restoration.utils.default_parser import default_argument_parser
from speechtask.punctuation_restoration.utils.utility import print_arguments
from speechtask.punctuation_restoration.io.dataset import PuncDataset
from speechtask.punctuation_restoration.model.lstm import RnnLm

import yaml

def main_sp(config, args):
    # print("Load datasets...")
    # # used for training
    # train_set = PRDataset(config["dataset"], shuffle=True)
    # # used for computing validate loss
    # valid_data = PRDataset(config["dev_set"], batch_size=1000, shuffle=True)[0]
    # valid_text = config["dev_text"]
    # test_texts = [config["ref_text"], config["asr_text"]]

    # print("Build models...")
    # model = RnnLm(config["model"])
    # model.train(train_set, valid_data, valid_text, test_texts)

    exp = Trainer(config, args)
    exp.setup()
    exp.run()


def main(config, args):
    if args.device == "gpu" and args.nprocs > 1:
        dist.spawn(main_sp, args=(config, args), nprocs=args.nprocs)
    else:
        main_sp(config, args)


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    print_arguments(args, globals())

    # https://yaml.org/type/float.html
    with open(args.config,"r") as f:
        config= yaml.load(f,Loader=yaml.FullLoader)

    print(config)
    if args.dump_config:
        with open(args.dump_config, 'w') as f:
            print(config, file=f)

    main(config, args)
