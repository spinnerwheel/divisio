import numpy as np 
import math
import skimage
import cv2
from skimage.feature import hog

def compute_local_descriptor(image, t_size, t_step, functions): 
    rows = 0 
    cols = 0
    function_out = []
    out = []  
    pad = math.floor(t_size/2)  
    #image = np.pad(image,((pad,pad),(pad,pad),(0,0)),'constant', constant_values=0)
    image = np.pad(image,((pad,pad),(pad,pad),(0,0)),'edge')
    [x,y,z] = image.shape  
    for i in range(pad,x-pad,t_step): 
        rows+=1 
        cols = 0
        for j in range(pad,y-pad,t_step):  
            cols+=1 
            textel = image[i:i+t_size,j:j+t_size,:] 
            for function in functions:  
                function_results = function(textel)
                for result in function_results: 
                    result = round(result,3)
                    function_out.append(result)
            out.append(function_out)
            function_out = []
    return out, rows, cols 

def get_mean(img): 
    r = img[:,:,0] 
    g = img[:,:,1] 
    b = img[:,:,2] 
    meanR = np.average(r) 
    meanG = np.average(g) 
    meanB = np.average(b) 
    return [meanR, meanG, meanB] 
 
def get_stdev(img): 
    r = img[:,:,0] 
    g = img[:,:,1] 
    b = img[:,:,2] 
    stdevR = np.std(r) 
    stdevG = np.std(g) 
    stdevB = np.std(b) 
    return [stdevR, stdevG, stdevB]

#Il Number of circularly symmetric neighbor set points (quantization of the angular space)  
# è uno dei parametri utilizzati nel calcolo del descrittore di texture LBP (Local Binary Pattern),  
#che nel nostro caso è il secondo parametro.  
# Questo parametro fa riferimento al numero di punti di campionamento che vengono distribuiti su un cerchio immaginario  
# intorno ad ogni pixel dell'immagine. La distribuzione di questi punti di campionamento può essere uniforme o non uniforme sulla circonferenza, 
# e il numero di punti di campionamento influisce sulla sensibilità del descrittore ai cambiamenti di texture.  
# In pratica, aumentando il numero di punti di campionamento si ottengono descrittori di texture più discriminanti,  
# ma questo aumenta anche il costo computazionale. 
# il secondo parametro è il raggio del cerchio 
# il terzo parametro è il metodo di calcolo del LBP(per altre info vedere la documentazione di skimage)

def get_LBP(img):
    img = skimage.color.rgb2gray(img)
    img = skimage.img_as_ubyte(img)
    lbp = skimage.feature.local_binary_pattern(img, 8, 1, method='uniform')
    lbp = lbp.tolist()
    hist, _ = np.histogram(lbp, bins=np.arange(0, 257))
    return hist.tolist()

def get_hog_features(patch):
    # patch è il tassello di immagine di input
    # converte l'immagine in scala di grigi
    patch_gray = skimage.color.rgb2gray(patch)
    # calcola le feature HOG
    fd, hog_image = hog(patch_gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, block_norm='L2-Hys')
    # restituisce la lista di feature HOG
    return fd.tolist()