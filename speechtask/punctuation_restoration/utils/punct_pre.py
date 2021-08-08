import os
import shutil

CHINESE_PUNCTUATION_MAPPING = {'O': '',
                                '，': "，",
                                '。': '。',
                                '？': '？',}

def process_one_file_chinese(raw_path,save_path):
    f = open(raw_path,'r', encoding='utf-8')
    save_file = open(save_path,'w', encoding='utf-8')
    for line in f.readlines():
        line=line.strip().replace(' ', '').replace(' ', '')
        for i in line:
            save_file.write(i+' ')
        save_file.write('\n')
    save_file.close()

def trans_vocab_chinese(raw_path,save_path):
    f = open(raw_path,'r', encoding='utf-8')
    save_file= open(save_path,'w', encoding='utf-8')
    for line in f.readlines():
        save_file.write(line[0]+'\n')
    save_file.close()


def process_chinese_pure_senetence(config):
    ####need raw_path, raw_train_file, raw_dev_file, raw_test_file, vocab_file, punc_file, save_path
    assert os.path.exists(os.path.join(config["raw_path"], config["raw_train_file"])) == True, "train file doesn't exist."
    assert os.path.exists(os.path.join(config["raw_path"], config["raw_dev_file"])) == True, "dev file doesn't exist."
    assert os.path.exists(os.path.join(config["raw_path"], config["raw_test_file"])) == True, "test file doesn't exist."
    assert os.path.exists(os.path.join(config["raw_path"], config["vocab_file"])) == True, "vocab file doesn't exist."
    assert os.path.exists(os.path.join(config["raw_path"], config["punc_file"])) == True, "punc file doesn't exist."

    train_file = os.path.join(config["raw_path"], config["raw_train_file"])
    dev_file = os.path.join(config["raw_path"], config["raw_dev_file"])
    test_file = os.path.join(config["raw_path"], config["raw_test_file"])
    if not os.path.exists(config["save_path"]):
        os.makedirs(config["save_path"])

    shutil.copy(os.path.join(config["raw_path"], config["vocab_file"]), os.path.join(config["save_path"], config["vocab_file"]))
    shutil.copy(os.path.join(config["raw_path"], config["punc_file"]), os.path.join(config["save_path"], config["punc_file"]))

    process_one_file_chinese(train_file, os.path.join(config["save_path"], "train"))
    process_one_file_chinese(dev_file, os.path.join(config["save_path"], "dev"))
    process_one_file_chinese(test_file, os.path.join(config["save_path"], "test"))
    # trans_vocab_chinese(os.path.join(config["raw_path"], config["vocab_file"]),os.path.join(config["save_path"], config["vocab_file"]))


def process_one_chinese_pair(raw_path,save_path):
    
    f=open(raw_path, 'r',encoding='utf-8')
    save_file = open(save_path, 'w', encoding='utf-8')
    for line in f.readlines():
        if(len(line.strip().split())==2):
            word, punc=line.strip().split()
            save_file.write(word+' '+CHINESE_PUNCTUATION_MAPPING[punc])
            if(punc=="。"):
                save_file.write("\n")
            else:
                save_file.write(" ")
    save_file.close()
    

def process_chinese_pair(config):
    ### need raw_path, raw_train_file, raw_dev_file, raw_test_file, punc_file, save_path
    assert os.path.exists(os.path.join(config["raw_path"], config["raw_train_file"])) == True, "train file doesn't exist."
    assert os.path.exists(os.path.join(config["raw_path"], config["raw_dev_file"])) == True, "dev file doesn't exist."
    assert os.path.exists(os.path.join(config["raw_path"], config["raw_test_file"])) == True, "test file doesn't exist."
    assert os.path.exists(os.path.join(config["raw_path"], config["punc_file"])) == True, "punc file doesn't exist."

    train_file = os.path.join(config["raw_path"], config["raw_train_file"])
    dev_file = os.path.join(config["raw_path"], config["raw_dev_file"])
    test_file = os.path.join(config["raw_path"], config["raw_test_file"])

    process_one_chinese_pair(train_file, os.path.join(config["save_path"], "train"))
    process_one_chinese_pair(dev_file, os.path.join(config["save_path"], "dev"))
    process_one_chinese_pair(test_file, os.path.join(config["save_path"], "test"))

    shutil.copy(os.path.join(config["raw_path"], config["punc_file"]), os.path.join(config["save_path"], config["punc_file"]))

