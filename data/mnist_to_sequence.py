"""
"""
import skimage.io as io 
import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
import os

def load_data(data_file: str):
    print('loading data ..')
    with open(data_file) as f:
        filenames = []
        labels = []
        for item in f:
            item = item.strip().split('\t')
            filenames.append(item[0])
            labels.append(int(item[1]))                        
    l_images = []
    for filename in filenames :
        image = io.imread(filename)
        l_images.append(image)
        
    data =np.stack(l_images, axis = 0)
    
    labels = np.array(labels)
    print(labels)
    return data, labels

def generate_sequence(digits, labels, seq_size):
    output_image = np.zeros((28,28*5), dtype = np.uint8)
    n_images = digits.shape[0]
    random_pos = random.randint(0,n_images, 5)
    val_string = '' 
    for j,i in enumerate(random_pos) :
        output_image[:, j*28:(j+1)*28] = data[i, :, :]
        val_string = val_string + str(labels[i])
    return (output_image, val_string)

if __name__ == '__main__' :
    data_file = '/home/vision/smb-datasets/MNIST/MNIST-5000/train.txt'
    np_file = 'mnist_data'     
#     data, labels = load_data(data_file)
#     np.save(np_file, data)
#     np.save(np_file+'_labels', labels)
#     print('OK')
    data = np.load(np_file+'.npy')
    labels = np.load(np_file+'_labels.npy')
    print(labels)
    n = 10
    for i in range(n) :
        output_image, val = generate_sequence(data, labels, 6)
        io.imsave(os.path.join('data_sample',val+'.png'), output_image)
        #plt.imshow(output_image, cmap = 'gray')        
        #plt.title(val)
        #plt.pause(1)
    #plt.show()
    
    
        