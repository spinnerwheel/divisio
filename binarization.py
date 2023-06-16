from utils import *
from skimage.morphology import disk


def is_empty(img):
    if cv2.countNonZero(img) == 0:
        return True
    else:
        return False
    
def images_binarization(write_path,read_path):
    images, names = load_images(read_path)
    image_results = [] 
    name_results =[]  
    i = 1

    for image,name in zip(images,names):
        length = len(images)
        print(f'Processing image: {i}/{length}', end='\r')
        im = image.copy()
        im = gray_scale(im)
        im = gaussian_blur(im,11)
        im = canny_edge(im,1.2,50,190)
        im = dilate_image(im,np.ones((2,2),np.uint8))
        im = multi_label_connected_components(im,350)
        im = dilate_image(im,disk(9))
        im = flood_filling(im,(0,0))
        if not is_empty(im):
            image_results.append(im)
            name_results.append(name)
        i+=1
    save_images(write_path,image_results,name_results)
    
def multi_images_binarization(images,names):
    ''' 
    images, names = load_images(read_path)
    ''' 
    multi_write_path = './multi_results/'
    image_results = [] 
    name_results =[] 

    i = 1

    for image,name in zip(images,names):
        length = len(images)
        print(f'Processing image: {i}/{length}', end='\r')
        im = image.copy()
        im = cv2.resize(im, (0,0), fx=0.25, fy=0.25)
        im = gray_scale(im)
        im = gaussian_blur(im,11)
        im = canny_edge(im,1.2,50,190)
        im = dilate_image(im,disk(2))
        multi_im = multi_label_connected_components(im,1000)
        j = 0
        for im in multi_im:
            im = dilate_image(im,disk(2))
            im = flood_filling(im,(0,0))
            if not is_empty(im):
                im = cv2.resize(im, (0,0), fx=2, fy=2)
                image_results.append(im)
                name = name.split('.')[0]
                name_results.append(f'{name}_{j}.png')
                j+=1
        i+=1
    save_images(multi_write_path,image_results,name_results)
    return image_results,name_results
    

if __name__ == '__main__':
    '''
    read_path = './named/'
    write_path = './results/'
    '''
    multi_read_path = './multi/'
    multi_write_path = './multi_results/'
    multi_images_binarization(multi_write_path,multi_read_path)