# -*- coding:utf-8 -*-
import numpy as np
import urllib
import urllib2
import requests
import cv2
import os
import glob
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import threading

row_size = 10
column_size = 10
Normal_value = 200
MINIMUM_SIZE = 4096

'''crawling method1'''
def download_data_from_internet_in_urllib(path,save_path):

    data = urllib.urlopen(path)
    '''information for url'''
    #print temp.geturl() #return url
    #print data.getcode() #return code
    #print data.info() # return header
    #print data.read() # return data
    #soup = BeautifulSoup(data.read(), 'lxml')
    #print soup

    '''
    Points to note when downloading url 'ImageNet' data.
    1. Using urllib.urlopen().getcode(), check the connection state - must be 200
    2. Using os.path.getsize(), Find image data that has a value less than 4096.
    3. Find data of the form 'None'.
    '''
    i=0
    for t,line in enumerate(data.readlines()):
        print "Image_URL -> "+line
        try :
            code=urllib.urlopen(line).getcode() # Normal value must be 200.
            print "code : {}".format(code)
            if code != Normal_value:
                raise  # Errors of the current except statement are indicated.
        except:
            '''not reading Image data that has a value that is not '200'. '''
            print "error_URL"+line
            i+=(-1)
            continue
        else:
            if code == Normal_value:
                print "download..."+"<"+save_path.format(t+1+i)+">"

                '''In here,because all Image data(including null data) is saved
                We have to do post-processing. '''
                urllib.urlretrieve(line,save_path.format(t+1+i))
                img = cv2.imread(save_path.format(t+1+i), cv2.IMREAD_COLOR)
                '''post-processing, removing image data that has a value less than 4096 and removing data of the form 'None' '''
                if img != None and os.path.getsize(save_path.format(t+1+i)) > MINIMUM_SIZE:
                    resized_image = cv2.resize(img, (64, 64),interpolation=cv2.INTER_AREA)
                    cv2.imwrite(save_path.format(t+1+i), resized_image)
                else:
                    print "Erase data that does not satisfy the condition."
                    os.remove(save_path.format(t+1+i))
                    i += (-1)
    print "download completed"

'''crawling method2'''
def download_data_from_internet_in_requests(path,save_path):

    data=requests.get(path,stream=True)

    '''
    Points to note when downloading url 'ImageNet' data.
    1. Using urllib.urlopen().getcode(), check the connection state - must be 200
    2. Using os.path.getsize(), Find image data that has a value less than 4096.
    3. Find data of the form 'None'.
    '''
    i=0
    for t,line in enumerate(data.iter_lines()):
        print "Image_URL -> " + line
        try:
            code = urllib.urlopen(line).getcode()  # Normal value must be 200.
            print "code : {}".format(code)
            if code != Normal_value:
                raise  # Errors of the current except statement are indicated.
        except:
            '''not reading Image data that has a value that is not '200'. '''
            print "error_URL" + line
            i += (-1)
            continue
        else:
            if code == Normal_value:
                print "download..." + "<" + save_path.format(t + 1 + i) + ">"

                '''In here,because all Image data(including null data) is saved
                We have to do post-processing. '''
                urllib.urlretrieve(line, save_path.format(t + 1 + i))
                img = cv2.imread(save_path.format(t + 1 + i), cv2.IMREAD_COLOR)
                '''post-processing, removing image data that has a value less than 4096 and removing data of the form 'None' '''
                if img != None and os.path.getsize(save_path.format(t + 1 + i)) > MINIMUM_SIZE:
                    resized_image = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(save_path.format(t + 1 + i), resized_image)
                else:
                    print "Erase data that does not satisfy the condition."
                    os.remove(save_path.format(t + 1 + i))
                    i += (-1)
    print "download completed"

def read_data_from_file():
    train_data=[]
    data_list=glob.glob("ImageNet/ImageNet_face/*")#return list of data_address

    '''1. fetch the data from 'ImageNet' folder'''
    train_data.append([cv2.imread(dl,cv2.IMREAD_COLOR) for dl in data_list])
    '''2. BGR - > RGB '''
    for i in xrange(len(train_data[0])):
        b,g,r=cv2.split(train_data[0][i])
        train_data[0][i]=cv2.merge([r,g,b])
    return np.asarray(train_data[0]).transpose(0,3,1,2)

if __name__ == "__main__":

    '''address contents'''
    url_face = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09618957'
    url_celebrity = 'http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n09889065'
    url_woman = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n09619168'

    '''ImageNet data'''
    path1 = "ImageNet/ImageNet_face/{}.jpg"
    path2 = "ImageNet/ImageNet_celebrity/{}.jpg"
    path3 = "ImageNet/ImageNet_woman/{}.jpg"

    download_data=False

####################################################################################
    if download_data==True:
        "download the ImageNet data from inside   , using thread"
        '''method1'''
        t1=threading.Thread(target=download_data_from_internet_in_urllib ,args=[url_celebrity,path2]) # using urllib1 or urllib2
        t2=threading.Thread(target=download_data_from_internet_in_requests ,args=[url_woman,path3])

        t1.start()
        t2.start()

### ####################################################################################
    else:
        print "Load the ImageNet data from inside"
        train_img=read_data_from_file()
        train_img=train_img.transpose(0,2,3,1)
        #visualization
        fig ,  ax = plt.subplots(row_size, column_size, figsize=(column_size, row_size))
        fig.suptitle('ImageNet')
        for j in xrange(row_size):
            for i in xrange(column_size):
                ax[j][i].set_axis_off()
                ax[j][i].imshow(train_img[i+j*column_size])
        plt.show()

else:
    print "Load the ImageNet data from the outside"



