import os
import random


# def _main():
#     val_percent = 0.1
#     train_percent = 0.9
#     xmlfilepath = 'trainrgb'
#     total_xml = os.listdir(xmlfilepath)
#
#     num = len(total_xml)
#     list = range(num)
#     tv = int(num * val_percent)
#     tr = int(tv * train_percent)
#     val = random.sample(list, tv)
#     train = random.sample(val, tr)
#
#     fval = open('txt/val.txt', 'w')
#     ftest = open('txt/test.txt', 'w')
#     ftrain = open('txt/train.txt', 'w')
#     ftestval = open('txt/val.txt', 'w')
#
#     for i in list:
#         name = '/trainrgb/'+total_xml[i][:-4] + '.png'+ ', ' + '/trainannot/'+total_xml[i][:-4] + '.png' + ', 0'+ '\n'
#         if i in val:
#             fval.write(name)
#             if i in train:
#                 ftest.write(name)
#             else:
#                 ftestval.write(name)
#         else:
#             ftrain.write(name)
#
#     fval.close()
#     ftrain.close()
#     ftestval.close()
#     ftest.close()

def _main():
    # val_percent = 0.1
    # train_percent = 0.9
    xmlfilepath = 'valrgb'
    total_xml = os.listdir(xmlfilepath)

    num = len(total_xml)
    list = range(num)
    # tv = int(num * val_percent)
    tv = int(num )
    # tr = int(tv * train_percent)
    # val = random.sample(list, tv)
    # train = random.sample(val, tr)
    val = random.sample(list,tv)

    fval = open('txt/val.txt', 'w')
    # ftest = open('txt/test.txt', 'w')
    # ftrain = open('txt/train.txt', 'w')
    # ftestval = open('txt/testval.txt', 'w')

    for i in list:
        name = '/valrgb/'+total_xml[i][:-4] + '.png'+ ', ' + '/valannot/'+total_xml[i][:-4] + '.png' + ', 0'+ '\n'
        if i in val:
            fval.write(name)
        #     if i in train:
        #         ftest.write(name)
        #     else:
        #         ftestval.write(name)
        # else:
        #     ftrain.write(name)

    fval.close()
    # ftrain.close()
    # ftestval.close()
    # ftest.close()
if __name__ == '__main__':
    _main()