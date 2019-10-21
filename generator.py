

import numpy as np
import cv2
import os
from tqdm import tqdm
import random
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import pickle
import glob 
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage
from shapely.geometry import Polygon


def show(image):

    scaleBg=iaa.Scale({"height": 480, "width": 480})
    #image = scaleBg.augment_image(image)
    cv2.imshow('Image' ,  image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def give_me_filename(dirname, suffixes, prefix=""):
    """
        Function that returns a filename or a list of filenames in directory 'dirname'
        that does not exist yet. If 'suffixes' is a list, one filename per suffix in 'suffixes':
        filename = dirname + "/" + prefix + random number + "." + suffix
        Same random number for all the file name
        Ex: 
        > give_me_filename("dir","jpg", prefix="prefix")
        'dir/prefix408290659.jpg'
        > give_me_filename("dir",["jpg","xml"])
        ['dir/877739594.jpg', 'dir/877739594.xml']        
    """
    if not isinstance(suffixes, list):
        suffixes=[suffixes]
    
    suffixes=[p if p[0]=='.' else '.'+p for p in suffixes]
          
    while True:
        bname="%09d"%random.randint(0,999999999)
        fnames=[]
        for suffix in suffixes:
            fname=os.path.join(dirname,prefix+bname+suffix)
            if not os.path.isfile(fname):
                fnames.append(fname)
                
        if len(fnames) == len(suffixes): break
    
    if len(fnames)==1:
        return fnames[0]
    else:
        return fnames



def display_img(img,polygons=[],channels="bgr",size=9):
 
    if not isinstance(polygons,list):
        polygons=[polygons]    
    if channels=="bgr": # bgr (cv2 image)
        nb_channels=img.shape[2]
        if nb_channels==4:
            img=cv2.cvtColor(img,cv2.COLOR_BGRA2RGBA)
        else:
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)    
    fig,ax=plt.subplots(figsize=(size,size))
    ax.set_facecolor((0,0,0))
    ax.imshow(img)
    for polygon in polygons:
        # An polygon has either shape (n,2), 
        # either (n,1,2) if it is a cv2 contour (like convex hull).
        # In the latter case, reshape in (n,2)
        if len(polygon.shape)==3:
            polygon=polygon.reshape(-1,2)
        patch=patches.Polygon(polygon,linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(patch)


 
 
def kps_to_BB(kps):
    """
        Determine imgaug bounding box from imgaug keypoints
    """
    extend=0 # To make the bounding box a little bit bigger
    kpsx=[kp.x for kp in kps.keypoints]
    minx=max(0,int(min(kpsx)-extend))
    maxx=min(imgW,int(max(kpsx)+extend))
    kpsy=[kp.y for kp in kps.keypoints]
    miny=max(0,int(min(kpsy)-extend))
    maxy=min(imgH,int(max(kpsy)+extend))
    if minx==maxx or miny==maxy:
        return None
    else:
        return ia.BoundingBox(x1=minx,y1=miny,x2=maxx,y2=maxy)

 
 
# imgW,imgH: dimensions of the generated dataset images 
imgW=1080
imgH=1080

# imgaug transformation for the background
scaleBg=iaa.Scale({"height": imgH, "width": imgW})

def augment(img, list_kps, seq, restart=True):
 

    # image_before = img
    # for kp in list_kps:
    #     image_before = kp.draw_on_image(image_before, size=7)
    # show(image_before)

    img_aug = seq.augment_images([img])[0]
    list_kps_aug = [seq.augment_keypoints([kp])[0] for kp in list_kps]


    # image_after = img_aug
    # for kp in list_kps_aug:
    #     image_after = kp.draw_on_image(image_after, size=7)
    # show(image_after)



    list_bbs=[]
    list_bbs=[kps_to_BB(kps_aug) for kps_aug in list_kps_aug]
 
    return img_aug,list_kps_aug,list_bbs

def map_min_max(x, in_min, in_max, out_min, out_max):
    return int((x-in_min) * (out_max-out_min) / (in_max-in_min) + out_min)

def apply_brightness_contrast(input_img, brightness = 255, contrast = 127):
    brightness = map_min_max(brightness, 0, 510, -255, 255)
    contrast = map_min_max(contrast, 0, 254, -127, 127)
 
    #print("brightness ",brightness)
    #print("contrast ", contrast)

    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
 
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
 
    if contrast != 0:
        f = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
 
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
 
    #cv2.putText(buf,'B:{},C:{}'.format(brightness,contrast),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return buf


class BBA:  # Bounding box + annotations
    def __init__(self,bb,classname):      
        self.x1=int(round(bb.x1))
        self.y1=int(round(bb.y1))
        self.x2=int(round(bb.x2))
        self.y2=int(round(bb.y2))
        self.classname=classname
 
    

class SceneCrate:
    def __init__(self,bg,top,middle,bottom, height):
        self.createScene(bg,top,middle,bottom, height)
  
        
    def createScene(self,bg,top,middle,bottom, height):


        self.list_kps = []
        self.final = bg
            
        
        
        self.startX = np.random.randint(100,imgW - 420)
        self.startY = np.random.randint(800,imgH - 150)
        
 


        #crate_images = bottom
        #print("Bottom")
        self.add_level(bottom, height)
        #print("Middle")
        self.add_level(middle, height)
        #print("Top")
        self.add_level(top, height) 



        random_scale = np.random.uniform(low=0.8, high=1.1, size=1)[0]
        random_translate = np.random.uniform(low=0, high=0.25, size=1)[0]
        random_flip = np.random.randint(0,2,size=1, dtype=np.uint8)
        random_rotate = np.random.randint(-8,8)
        reduce_height = random_rotate

        # seq = iaa.Sequential([
        #     iaa.Affine(
        #         rotate=np.random.randint(0,10),
        #         scale=random_scale,
        #         translate_percent={"x":(-random_translate,random_translate),"y":(-random_translate,random_translate)},
        #     ) # rotate by exactly 10deg and scale to 50-70%, affects keypoints
        # ])


        transform_seq = iaa.Sequential([
            iaa.Affine(scale=random_scale),
            iaa.Affine(rotate=random_rotate),
            iaa.Fliplr(random_flip),
            iaa.Sometimes(0.5,
                iaa.GaussianBlur(sigma=(0, 0.5))
            ),
            #iaa.Affine(translate_percent={"x":(-random_translate,random_translate),"y":(-random_translate,random_translate)}),
            #iaa.PerspectiveTransform(scale=0.1,keep_size=True)
        ])

        self.img_augmented,self.lkps,self.bbs=augment(self.final,self.list_kps,transform_seq)
        self.listbba = []
        for bb in self.bbs:
            self.listbba.append(BBA(bb,"crate")) 
     
        
    def add_level(self,crate_images, height): 
        n=0

        num_crates_on_level =  np.random.randint(2,6)
        for i in range(num_crates_on_level):

             
            crate = np.copy(crate_images[n])
 

            if n == len(crate_images)-1:
                n =0
            else:
                n= n + 1
             
            crate_height, crate_width = crate.shape[:2]

            self.startY=self.startY - height

            if self.startY < 40:
                break
            
            
            kps = KeypointsOnImage([
                Keypoint(x=self.startX, y=self.startY),
                Keypoint(x=self.startX + crate_width, y=self.startY),
                Keypoint(x=self.startX + crate_width, y=self.startY + crate_height),
                Keypoint(x=self.startX, y=self.startY + crate_height)
            ], shape=(imgH,imgW,3))

            self.list_kps.append(kps)
 
        
            brightness_delta = np.random.randint(0, 100)
            contrast_delta = np.random.randint(-30, 30)
            crate[:,:,:3] = apply_brightness_contrast(crate[:,:,:3], 255-brightness_delta, 127-contrast_delta)
        
 
        
            self.img=np.zeros((imgH,imgW,4),dtype=np.uint8)
            self.img[self.startY:self.startY + crate_height, self.startX:self.startX + crate_width,:]=crate 
            

            
            # self.class1=class1
            # self.class2=class2
            # self.class3=class3
            
            
            # Construct final image of the scene by superimposing: bg, img1, img2 and img3
             
 
            mask=self.img[:,:,3]
            self.mask=np.stack([mask]*3,-1)
            self.final=np.where(self.mask,self.img[:,:,0:3],self.final)
 

            #show(self.final)
            #display_img(self.final,polygons=[],channels="bgr",size=9)



    def display(self):
        fig,ax=plt.subplots(1,figsize=(8,8))
        ax.imshow(self.final)
        for bb in self.listbba:
            rect=patches.Rectangle((bb.x1,bb.y1),bb.x2-bb.x1,bb.y2-bb.y1,linewidth=1,edgecolor='b',facecolor='none')
            ax.add_patch(rect)
    def res(self):
        return self.final
    def write_files(self,save_dir,display=False):
        jpg_fn, xml_fn=give_me_filename(save_dir, ["jpg","xml"])
        plt.imsave(jpg_fn,self.img_augmented)
        if display: print("New image saved in",jpg_fn)
        create_voc_xml(xml_fn,jpg_fn, self.listbba,display=display)


# Pickle file containing the background images from the DTD

class Backgrounds():
    def __init__(self,backgrounds_pck_fn):
        self._images=pickle.load(open(backgrounds_pck_fn,'rb'))
        self._nb_images=len(self._images)
        print("Nb of images loaded :", self._nb_images)
    def get_random(self, display=False):
        bg=self._images[random.randint(0,self._nb_images-1)]
        if display: plt.imshow(bg)
        return bg
    

xml_body_1="""<annotation>
        <folder>FOLDER</folder>
        <filename>{FILENAME}</filename>
        <path>{PATH}</path>
        <source>
                <database>Unknown</database>
        </source>
        <size>
                <width>{WIDTH}</width>
                <height>{HEIGHT}</height>
                <depth>3</depth>
        </size>
"""
xml_object=""" <object>
                <name>{CLASS}</name>
                <pose>Unspecified</pose>
                <truncated>0</truncated>
                <difficult>0</difficult>
                <bndbox>
                        <xmin>{XMIN}</xmin>
                        <ymin>{YMIN}</ymin>
                        <xmax>{XMAX}</xmax>
                        <ymax>{YMAX}</ymax>
                </bndbox>
        </object>
"""
xml_body_2="""</annotation>        
"""

def create_voc_xml(xml_file, img_file,listbba,display=False):
    with open(xml_file,"w") as f:
        f.write(xml_body_1.format(**{'FILENAME':os.path.basename(img_file), 'PATH':img_file,'WIDTH':imgW,'HEIGHT':imgH}))
        for bba in listbba:            
            f.write(xml_object.format(**{'CLASS':bba.classname,'XMIN':bba.x1,'YMIN':bba.y1,'XMAX':bba.x2,'YMAX':bba.y2}))
        f.write(xml_body_2)
        if display: print("New xml",xml_file)



#-----------------------------------------------------------------------------

data_dir="data" # Directory that will contain all kinds of data (the data we download and the data we generate)
#save_dir="data/scenes/train"
save_dir="data/scenes/val"

if not os.path.isdir(data_dir):
    os.makedirs(data_dir)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


counter = 0

backgrounds_pck_fn="data/backgrounds.pck"
backgrounds = Backgrounds(backgrounds_pck_fn)

classes = [ "vid", 
            "vid2",
            "vid3side",
            "crate4"
            
            ]
class_height = [76,78,67,71]
images_per_class = 50
 


crates_files = glob.glob('data/crates/*')
   
for obj_idx, obj_class in enumerate(classes):
    obj_class = "data/crates/" + obj_class #'data/crates/crete4'

    crate_pos_dirs = glob.glob(obj_class + '/*')
    print('Processing: ', obj_class)
    class_images = []

    #traverse top,middle,bottom folders
    crate_top_dir = glob.glob(obj_class + '/top/*')
    crate_top = []

    print("Top dir", crate_top_dir)

    for tl_idx, fname in enumerate(crate_top_dir):           
        print("Loading:",fname)
        image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        crate_top.append(image)

    crate_middle_dir = glob.glob(obj_class + '/mid/*')
    crate_middle = []

    for tl_idx, fname in enumerate(crate_middle_dir):           
        print("Loading:",fname)
        image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        crate_middle.append(image)

    crate_bottom_dir = glob.glob(obj_class + '/bot/*')
    crate_bottom = []
    
    for tl_idx, fname in enumerate(crate_bottom_dir):           
        print("Loading:",fname)
        image = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
        crate_bottom.append(image)


    for num_img in range(images_per_class):
        bg=backgrounds.get_random()
        bg=scaleBg.augment_image(bg)
        #bg=image
        newimg=SceneCrate(bg,crate_top, crate_middle, crate_bottom , class_height[obj_idx])  
        newimg.write_files(save_dir)
        counter +=1

        print(counter)


        #draw bounding boxes
        # colors = [(255,0,0),(255,255,0),(255,0,255),(0,255,0),(0,0,255),(0,255,255),(255,0,0),(255,255,0),(255,0,255),(0,255,0),(0,0,255),(0,255,255),(255,0,0),(255,255,0),(255,0,255),(0,255,0),(0,0,255),(0,255,255),]
        # i=0

        # result_img = np.copy(newimg.img_augmented)

        # for bb in newimg.bbs:
        #     cv2.rectangle(result_img, (bb.x1_int, bb.y1_int), (bb.x2_int, bb.y2_int), colors[i], 2)
        #     #show(result_img)
        #     i+=1

        # show(result_img)

        #newimg.display()    
 

 #converter fr yolo
 #python3 convert_voc_yolo.py data/scenes/train data/crates.names data/train.txt