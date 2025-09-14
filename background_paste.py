import os
import albumentations as A
import cv2
from PIL import Image
import random


def scale_image(image):
    # use scaling on image.size
    uniform_scale = random.uniform(0.1, .7)
    img_h = image.shape[0] * uniform_scale
    img_w = image.shape[1] * uniform_scale
    image = cv2.resize(image, (int(img_w), int(img_h)))

    return image, uniform_scale

def rotate_image(image, bboxes):

    transform = A.Compose([
        A.Rotate(limit=25, p=1, border_mode=cv2.BORDER_CONSTANT),
    ], bbox_params=A.BboxParams(format='yolo'))
    transformed = transform(image=image, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']

    return transformed_image, transformed_bboxes

def move_bbox(bboxes, pose, ov_h, ov_w, bg_h, bg_w):
    x_offset, y_offset=pose

    x_cent = bboxes[0][0]
    y_cent = bboxes[0][1]
    #bbox_w = bboxes[0][2]
    #bbox_h = bboxes[0][3]

    abs_x_cent = x_offset + x_cent * ov_h
    abs_y_cent = y_offset + y_cent * ov_w
    # abs_bbox_w = bbox_w * ov_w
    # abs_bbox_h = bbox_h * ov_h

    new_x = abs_x_cent / bg_w
    new_y = abs_y_cent / bg_h
    # new_bbox_w = abs_bbox_w / bg_w
    # new_bbox_h = abs_bbox_h / bg_h

    return new_x, new_y

file = os.listdir('/Image_url')
for i in file:
    #read image
    image=cv2.imread(f'/Image_url/{i[:-3]}png', cv2.IMREAD_UNCHANGED)

    #read label
    with open(f'/Label_url/{i[:-3]}txt', 'r') as f:
        bboxes = [list(map(float, line.strip().split())) for line in f.readlines()]

    #set label into the Albumentation format
    bboxes=[[bboxes[0][1], bboxes[0][2], bboxes[0][3], bboxes[0][4],'0']]

    #scale the image
    image, uniform_scale = scale_image(image)
    ov_w = image.shape[0]
    ov_h = image.shape[1]

    #set background
    background = Image.open('/background_image/0.jpg')
    bg_w = background.size[0]
    bg_h = background.size[1]

    #set a random a position for the overlay on the background
    x = random.randint(0, 900)
    y = random.randint(500, 900)
    pose=(x,y)

    #bboxes= [[bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3],'0'],]
    #rotate the image with it's label coordinate
    transformed_image, transformed_bboxes= rotate_image(image, bboxes)

    bboxes = transformed_bboxes


    #move the label to the right place on the background
    new_x, new_y = move_bbox(transformed_bboxes, pose, ov_h, ov_w, bg_h, bg_w)

    #normalazie the yolo label ('class_id, x, y, width, height)
    bboxes=['0',str(new_x),str(new_y),str(transformed_bboxes[0][2]*uniform_scale),str(transformed_bboxes[0][3]*uniform_scale)]
    print(bboxes)

    #change the image format into the PIL type
    image= Image.fromarray(transformed_image)

    #paste the image on background
    background.paste(image, pose, image)

    #save the image and label
    background.save(f'/image_results/{i}.jpg')
    final_label = open(f'/label_results/{i}.txt', 'w')
    final_label.write(' '.join(bboxes) + '\n')

#cv2.imwrite('/url', transformed_image)

