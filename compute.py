import numpy as np 
import math
import skimage

def compute_local_descriptor(image, t_size, t_step, functions): 
    rows = 0 
    cols = 0
    function_out = []  
    out = []  
    pad = math.floor(t_size/2)  
    image = np.pad(image,((pad,pad),(pad,pad),(0,0)),'constant',constant_values=0)  
    [x,y,z] = image.shape  
    for i in range(pad,x-pad,t_step): 
        rows+=1 
        cols = 0
        for j in range(pad,y-pad,t_step):  
            cols+=1 
            textel = image[i:i+t_size,j:j+t_size,:] 
            for function in functions:  
                function_out.append(function(textel)) 
            out.append(list(np.concatenate(function_out).flat)) 
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
    lbp = skimage.feature.local_binary_pattern(img, 8, 1, method='default') 
    return lbp
