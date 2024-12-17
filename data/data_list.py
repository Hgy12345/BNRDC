import json
import os
def load_file_list_recursion(fpath, result):
    allfilelist = os.listdir(fpath)
    for file in allfilelist:
        filepath = os.path.join(fpath, file)
        if os.path.isdir(filepath):
            load_file_list_recursion(filepath, result)
        else:
            result.append(filepath)
            print(len(result))



def scan(input_path, out_put):
    result_list = []
    load_file_list_recursion(input_path, result_list)
    result_list.sort()

    for i in range(len(result_list)):
        print('{}_{}'.format(i, result_list[i]))

    with open(out_put, 'w') as j:
        json.dump(result_list, j)

# scan(r'D:\A_image_inpainting\code\fwq\misf-main-xg2-11\test', './yl.txt')
# # scan(r'D:\datasets\celebA\dval256', './val.txt')
scan(r'D:\datasets\celebA\dtest256', './test.txt')
# scan(r'D:\A_image_inpainting\code\dataset\mask_divide\test_40', './mask.txt')