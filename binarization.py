from utils import *
from better_bin import *
from skimage.morphology import disk


def is_empty(img):
    if cv2.countNonZero(img) == 0:
        return True
    else:
        return False

if __name__ == '__main__':
    images, names = load_images('./named/')
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
        im = label_connected_components(im,350)
        im = dilate_image(im,disk(9))
        im = flood_filling(im,(0,0))

        image_results.append(im)
        name_results.append(name)
        i+=1
    save_images('./results/',image_results,name_results)