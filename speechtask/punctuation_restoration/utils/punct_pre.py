import os
import shutil


def process_one_file_chinese(raw_data,save_path):
    save_file= open(save_path,'w', encoding='utf-8')
    for line in raw_data.readlines():
        line=line.strip().replace(' ', '').replace(' ', '')
        for i in line:
            save_file.write(i+' ')
        save_file.write('\n')
    save_file.close()

def process_vocab_chinese(raw_data,save_path):
    f = open(raw_data,'r', encoding='utf-8')
    save_file= open(save_path,'w', encoding='utf-8')
    for line in f.readlines():
        save_file.write(line[0]+'\n')
    save_file.close()


def process_chinese(config):
    assert os.path.exists(os.path.join(config["raw_path"], config["raw_train_file"])) == True, "train file doesn't exist."
    assert os.path.exists(os.path.join(config["raw_path"], config["raw_dev_file"])) == True, "dev file doesn't exist."
    assert os.path.exists(os.path.join(config["raw_path"], config["raw_test_file"])) == True, "test file doesn't exist."
    assert os.path.exists(os.path.join(config["raw_path"], config["vocab_file"])) == True, "vocab file doesn't exist."
    assert os.path.exists(os.path.join(config["raw_path"], config["punc_file"])) == True, "punc file doesn't exist."

    train_file = open(os.path.join(config["raw_path"], config["raw_train_file"]),'r', encoding='utf-8')
    dev_file = open(os.path.join(config["raw_path"], config["raw_dev_file"]),'r', encoding='utf-8')
    test_file = open(os.path.join(config["raw_path"], config["raw_test_file"]),'r', encoding='utf-8')
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    shutil.copy(os.path.join(config["raw_path"], config["vocab_file"]), os.path.join(config["save_path"], config["vocab_file"]))
    shutil.copy(os.path.join(config["raw_path"], config["punc_file"]), os.path.join(config["save_path"], config["punc_file"]))

    process_one_file_chinese(train_file, os.path.join(config["save_path"], "train"))
    process_one_file_chinese(dev_file, os.path.join(config["save_path"], "dev"))
    process_one_file_chinese(test_file, os.path.join(config["save_path"], "test"))
    # process_vocab_chinese(os.path.join(config["raw_path"], config["vocab_file"]),os.path.join(config["save_path"], config["vocab_file"]))