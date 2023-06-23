import cv2
from segment_anything import SamAutomaticMaskGenerator
import os
import torch
from segment_anything import sam_model_registry
import supervision as sv
from time import process_time
import gc
torch.cuda.memory_summary(device=None, abbreviated=False)
import torch.utils.checkpoint as checkpoint
from torch.utils.checkpoint import checkpoint_sequential
os.environ['PYTORCH_CUDA_ALLOC_CONF']='max_split_size_mb:512' #'garbage_collection_threshold:0.8,max_split_size_mb:512'

gc.collect()
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.get_device_name(0))
print(torch.cuda.mem_get_info())
HOME = os.getcwd()
print("HOME:", HOME)
DEVICE = torch.device("cuda:0")
#DEVICE = 1
MODEL_TYPE = "vit_h"
print('DEVICE  : ',DEVICE)

# font
font = cv2.FONT_HERSHEY_SIMPLEX
# org
org = (50, 100)
# fontScale
fontScale = 1
# Blue color in BGR
color = (255, 0, 0)
# Line thickness of 2 px
thickness = 5
# Using cv2.putText() method

CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH))
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam, points_per_batch=32)
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")

IMAGE_PATH = ''
IMAGE_PATH = os.path.join(HOME, IMAGE_PATH)
print(IMAGE_PATH)

# Start the stopwatch / counter 
t1_start = process_time()

img = cv2.imread(IMAGE_PATH)

print('Original Dimensions : ',img.shape)

scale_percent = 100 # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

# resize image
image_bgr = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
print(' resize image')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
sam_result = mask_generator.generate(image_rgb)

print(len(sam_result))
for i in sam_result:
 x1,y1,w,h = i["bbox"]
 
 #print(x1,y1,w,h)
 area = i["area"]
 image_bgr = cv2.rectangle(image_rgb, (int(x1), int(y1)),  (int(x1)+int(w), int(y1)+int(h)), (255, 0, 0), 1)
 #image_rgb  = cv2.putText( image_rgb ,"Area = " + str(area), (x1 ,y1), font, 
  #                 1, color, 1, cv2.LINE_AA)
#cv2.imshow("Test1",image_rgb)

mask_annotator = sv.MaskAnnotator()
print(' mask_annotator',mask_annotator)
detections = sv.Detections.from_sam(sam_result=sam_result)
print(detections[0])

del sam_result
annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)
print('annotated_image',annotated_image)
# Stop the stopwatch / counter
t1_stop = process_time()   
print("Elapsed time:", t1_stop, t1_start)
print("Elapsed time during the whole program in seconds:",t1_stop-t1_start)


annotated_image = cv2.putText(annotated_image,"Process Time = " + str(t1_stop-t1_start), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)


cv2.imshow("Test",annotated_image)
sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)

