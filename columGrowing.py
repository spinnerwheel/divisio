import numpy as np
from PIL import Image

class columGrowing:
    image = None
    output_image = None
    inside = False
    
    def __init__(self,image):
        self.image = image
        self.output_image = np.zeros((image.shape[0],image.shape[1])).astype(np.uint8)
        self.base = image.shape[0]
        self.height = image.shape[1]
        self.inside = False
        self.growing_image = image.copy()
    
    def _keep(self,row, colum, a,b):
        '''
        a = valore da salvare in immagine,
        b = valore da salvare in output_image
        '''
        while row < self.base and self.image[row,colum] == a:
            self.output_image[row,colum] = b
            row += 1
        return row
    
    def _isInside(self,row,colum):
        while row < self.base and self.image[row,colum] == 0:
            row += 1
        if row == self.height:
            return False
        else:
            return True
        
    def recursive_call(self):
        for colum in range(self.base):
            self.disp = 0
            self.inside = False
            row = 0
            while row < self.height:
                if self.image[row,colum] == 0:                  #trovo n zeri
                    row = self._keep(row,colum,0,0)
                if row < self.height and self.image[row,colum] == 255:      #trovo m 255 sto per entrare
                    row = self._keep(row,colum,255,255)
                if row < self.height and self.image[row,colum] == 0:        #trovo s zeri sono dentro o fuori
                    if self._isInside(row,colum):
                        self.inside = not self.inside
                        if self.inside:
                            #self.inside_growing(row,colum)
                            row = self._keep(row,colum,0,255)
                        else:
                            row = self._keep(row,colum,0,0)
                            self.inside = not self.inside
                    else:
                        row = self._keep(row,colum,0,0)
                        
    def recursive_call_two(self,row,colum):
        if row-1 >= 0 and self.image[row-1,colum] == 0 and self.growing_image[row-1,colum] != 2:
            self.inside_growing(row-1,colum)
        if row+1 < self.base and self.image[row+1,colum] == 0 and self.growing_image[row-1,colum] != 2:
            self.inside_growing(row+1,colum)
        if colum-1 >= 0 and self.image[row,colum-1] == 0 and self.growing_image[row-1,colum] != 2:
            self.inside_growing(row,colum-1)
        if colum+1 < self.height and self.image[row,colum+1] == 0 and self.growing_image[row-1,colum] != 2:
            self.inside_growing(row,colum+1)
                        
    def inside_growing(self,row,colum):
        self.growing_image[row,colum] = 2
        self.output_image[row,colum] = 255
        self.recursive_call_two(row,colum)
            
                    
if __name__ == '__main__':
    image = np.array([[0,255,255,255,255,255,255,255,255,0],
                      [0,255,0,0,0,0,0,0,255,0],
                      [0,255,0,255,255,255,255,0,255,0],
                      [0,255,0,255,0,0,255,0,255,0],
                      [0,255,0,255,0,0,255,0,255,0],
                      [0,255,0,255,0,0,255,0,255,0],
                      [0,255,0,255,255,255,255,0,255,0],
                      [0,255,0,0,0,0,0,0,255,0],
                      [0,255,255,255,255,255,255,255,255,0],
                      [0,0,0,0,0,0,0,0,0,0]])
    a = image.copy()
    region = columGrowing(image)
    region.recursive_call()
    out = np.concatenate((a,region.output_image),axis=1)
    im = Image.fromarray(out)
    im.show()
    
                    