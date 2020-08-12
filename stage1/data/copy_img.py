import os
import argparse
import random
import shutil
from shutil import copyfile



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        # print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    # print('Create path - %s'%dir_path)


def main(config):
    rm_mkdir(config.trainrgb)
    rm_mkdir(config.trainannot)

    rm_mkdir(config.validannot)
    rm_mkdir(config.validrgb)

    # rm_mkdir(config.test_path)
    # rm_mkdir(config.test_GT_path)

    filenames = os.listdir(config.origin_data_path)
    data_list = []
    GT_list = []

    for filename in filenames:
        ext = os.path.splitext(filename)[-1]  # 提取文件名的后缀
        if ext == '.png':
            filename = filename.split('_')[-1][:-len('.png')]
            data_list.append(filename + '.png')
            GT_list.append(filename + '.png')

    num_total = len(data_list)
    num_train = int((config.train_ratio / (config.train_ratio + config.valid_ratio )) * num_total)
    num_valid = int((config.valid_ratio / (config.train_ratio + config.valid_ratio )) * num_total)
    # num_test = num_total - num_train - num_valid

    print('\nNum of train set : ', num_train)
    print('\nNum of valid set : ', num_valid)
    # print('\nNum of test set : ', num_test)

    Arange = list(range(num_total))
    random.shuffle(Arange)

    for i in range(num_train):
        idx = Arange.pop()
        # 将原始的所有数据选取和train相同的数据copy
        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.trainrgb, data_list[idx])
        copyfile(src, dst)
        # 将原始的所有数据选取和train-GT相同的数据copy
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.trainannot, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_train, prefix='Producing train set:', suffix='Complete', length=50)

    for i in range(num_valid):
        idx = Arange.pop()
        # 将原始的所有数据选取和valid相同的数据copy
        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.validrgb, data_list[idx])
        copyfile(src, dst)
        # 将原始的所有数据选取和valid-GT相同的数据copy
        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.validannot, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_valid, prefix='Producing valid set:', suffix='Complete', length=50)

    # for i in range(num_test):
    #     idx = Arange.pop()
    #     # 将原始的所有数据选取和test相同的数据copy
    #     src = os.path.join(config.origin_data_path, data_list[idx])
    #     dst = os.path.join(config.test_path, data_list[idx])
    #     copyfile(src, dst)
    #     # 将原始的所有数据选取和test-GT相同的数据copy
    #     src = os.path.join(config.origin_GT_path, GT_list[idx])
    #     dst = os.path.join(config.test_GT_path, GT_list[idx])
    #     copyfile(src, dst)
    #
    #     printProgressBar(i + 1, num_test, prefix='Producing test set:', suffix='Complete', length=50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # model hyper-parameters
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--valid_ratio', type=float, default=0.1)
    # parser.add_argument('--test_ratio', type=float, default=0.05)

    # data path
    parser.add_argument('--origin_data_path', type=str,
                        default='/home/zhouyilong/dataset/2/img')
    parser.add_argument('--origin_GT_path', type=str,
                        default='/home/zhouyilong/dataset/2/groundtruth')

    parser.add_argument('--trainrgb', type=str, default='/home/zhouyilong/github/YNet-master/stage1/data/trainrgb')
    parser.add_argument('--trainannot', type=str, default='/home/zhouyilong/github/YNet-master/stage1/data/trainannot')
    parser.add_argument('--validrgb', type=str, default='/home/zhouyilong/github/YNet-master/stage1/data/valrgb')
    parser.add_argument('--validannot', type=str, default='/home/zhouyilong/github/YNet-master/stage1/data/valannot')
    # parser.add_argument('--test_path', type=str, default='/home/zhouyilong/github/R2AttU_Net/data/test/')
    # parser.add_argument('--test_GT_path', type=str, default='/home/zhouyilong/github/R2AttU_Net/data/test_GT/')

    config = parser.parse_args()
    print(config)
    main(config)