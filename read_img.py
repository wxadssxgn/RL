import cv2

def readData(path, h=size, w=size):
    '''
    本函数主要实现图片的读取以及大小的修改和调整
    :param path:图片存储路径
    :param h:图片的大小
    :param w:同上
    '''
    # i = 1
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename1 = path + '/' + filename

            img = cv2.imread(filename1)

            top, bottom, left, right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            img = cv2.resize(img, (h, w))
            prefix = filename.split('.')[0]
            # print(str(prefix).isdigit())
            if str(prefix).isdigit():
                imgs.append(img)
                labs.append(path)
            else:
                name = prefix.split('_')[1]
                imgs.append(img)
                labs.append(name)
