"""
Created on Tue Oct 19 12:16:19 2024

@author: boazs
"""

## ======= WATERMARK IMAGES =======
    
    
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import bisect

## Paths
input_folder = r"path\TO_BE_WATERMARKED\NOT_watermarked" # take images from here   
output_folder =  r"path\TO_BE_WATERMARKED\watermarked_with_py" # place the watermarked here 

# params for randomizing watermarks:
opac_range= np.linspace(0.05, 0.6, 30),
opac_range = [np.round(x,2) for x in opac_range]
gen_params = {
          'text_length': {},
          'char_bag': 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZABCDEFGHIJKLMNOPQRSTUVWXYZ       !@-', 
          'font_bag': 6, # the number of fonts to chose from in openCV
          'scale_range': {},
          'scale_range_signature': {},
          'thick_range': {},
          'thick_range_signature': {},
          'opac_range': opac_range,
          'color': ['w', 'w', 'w', 'o', 'o'],  # o is other, random. Most cases will be white
          'repeated_mark': [1, 1, 1, 1, 1, 2, 2, 3, 4], # not yet implemented 
          'bottom_signature_rate': 20, # in percent (not ini ratio) the chances that an image will have a WM at the lower edge (and with a narrower range of width values) instead of the above 
          }
gen_params['text_length'][400] = [6, 14]
gen_params['text_length'][10000] = [8, 20]
gen_params['thick_range'][400] = [2, 4]
gen_params['thick_range'][750] = [3, 9]
gen_params['thick_range'][10000] = [3, 14]
gen_params['thick_range_signature'][400] = [2, 3]
gen_params['thick_range_signature'][750] = [2, 7]
gen_params['thick_range_signature'][10000] = [2, 10]
gen_params['scale_range'][400] = [2, 4]
gen_params['scale_range'][750] = [2, 5]
gen_params['scale_range'][10000] = [2, 8]
gen_params['scale_range_signature'][400] = [1, 3]
gen_params['scale_range_signature'][750] = [2, 4]
gen_params['scale_range_signature'][10000] = [2, 6]

             
def get_key_by_size(image_length, inner_dict):
    sorted_imSize = sorted(inner_dict.keys())
    index = bisect.bisect_left(sorted_imSize, image_length)
    return sorted_imSize[index]


def chose_rand_params(gen_params, shape=[500, 900]):
    params = dict()
    # generate a random text:
    key = get_key_by_size(shape[1], gen_params['text_length'])
    txt_len = int(np.random.randint(gen_params["text_length"][key][0], gen_params["text_length"][key][1], 1)[0])
    slct = np.array(range(len(gen_params['char_bag'])))
    np.random.shuffle(slct)
    slct = slct[:txt_len]
    gn = (gen_params['char_bag'][x] for x in slct)
    text = ''
    for i in range(txt_len):
        text = ''.join([text, next(gn)])
    params['text'] = text
    # generate some rand font parameters
    # First check if this is a reg watermark or a signature
    slctType = np.random.randint(0, 100)
    if slctType > gen_params['bottom_signature_rate']:
        scaleRange = gen_params['scale_range']
        thickRange = gen_params['thick_range']
        position_bounds_V = [50, round(shape[0]*0.92)]
    else:
        scaleRange = gen_params['scale_range_signature']
        thickRange = gen_params['thick_range_signature']
        position_bounds_V = [round(shape[0]*0.92), round(shape[0]*0.98)]
    params["font"] = np.random.randint(0,gen_params['font_bag'])
    key = get_key_by_size(shape[1], scaleRange)
    params["font_scale"] = np.random.randint(scaleRange[key][0], scaleRange[key][1])
    key = get_key_by_size(shape[1], thickRange)
    params["thickness"] = np.random.randint(thickRange[key][0], thickRange[key][1])
    opac_select = np.random.randint(0, len(gen_params["opac_range"][0])) 
    params["opacity"] = gen_params["opac_range"][0][opac_select]  
    clr_select = np.random.randint(0, len(gen_params['color']))
    params["color"] = gen_params["color"][clr_select]
    select_numb = np.random.randint(0, len(gen_params["repeated_mark"]))  
    params["repeated"] = gen_params["repeated_mark"][select_numb] # TBI
    position_bounds_H = [20, round(shape[1]/3)]
    params["start_left"] = np.random.randint(position_bounds_H[0], position_bounds_H[1])
    params["start_top"] = np.random.randint(position_bounds_V[0], position_bounds_V[1])  
    
    return params    

def create_watermark(params, shape=[500, 900]):
    # Create a frame for the watermark
    watermark = np.zeros((shape[0], shape[1], 4), dtype=np.uint8)
    # get the text
    text_size = cv2.getTextSize(params["text"], params["font"], params["font_scale"], params["thickness"])[0]
    text_x = (watermark.shape[1] - text_size[0]) // 2
    text_y = params["start_top"]  #  (watermark.shape[0] + text_size[1]) // 2
    # get the color
    colorDict = {'w': (255, 255, 255, 255),
                 'b': (0, 0, 0, 255), # black is not working properly
                 'o': (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256), 255)}
    color = colorDict[params["color"]]
    # Add text
    cv2.putText(watermark, params["text"], (text_x, text_y), params["font"], params["font_scale"], color, params["thickness"], cv2.LINE_AA)
    # Make the background transparent
    watermark[:, :, 3] = 0  # Set alpha channel to fully transparent
    watermark[:, :, 3] = (watermark[:, :, 0] > 0).astype(np.uint8) * int(255 * params["opacity"])

    return watermark


def apply_watermark(image, watermark):
    # position = [params["start_left"], params["start_top"]]
    overlay = image.copy()
    blended_rgb = cv2.addWeighted(overlay[:,:,:3], 1, watermark[:,:,:3], 0.3, 0)
    # blend the alpha channel
    alpha_overlay = overlay[:,:,3]
    alpha_watermark = watermark[:, :, 3]
    blended_alpha = cv2.addWeighted(alpha_overlay, 1, alpha_watermark, 0.3, 0)
    # join together 
    overlay[:, :, :3] = blended_rgb
    overlay[:, :, 3] = blended_alpha

    return overlay

# Watermark images in a folder

os.makedirs(output_folder, exist_ok=True) 
    
# RUN on a list of N random images within a folder 
Nfls = 0  # 10000 # N of images from the folder to be watermarked. If 0 take all files in folder
allFigs = os.listdir(input_folder)
np.random.shuffle(allFigs)
Nfls = (len(allFigs) if Nfls==0 else Nfls)
for imne in range(Nfls):
    image_name = allFigs[imne]
    if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
        img_path = os.path.join(input_folder, image_name)
        img = cv2.imread(img_path)
        if img.shape[2] != 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            # create a unique randomized watermark for the image
            params = chose_rand_params(gen_params, img.shape[:2])
            watermark = create_watermark(params, img.shape[:2])
            watermarked_img = apply_watermark(img, watermark)
            output_path = os.path.join(output_folder, image_name)
            cv2.imwrite(output_path, watermarked_img)
            cv2.destroyAllWindows()

# RUN A SINGLE FILE (in case you need just one...)

if False:
    
    input_folder = r"path\to\an\image.jpg"  
    output_folder = r"path\to\watermaked\watermarked_images"
    os.makedirs(output_folder, exist_ok=True)
    
    # load image
    img = cv2.imread(input_folder)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    image = img
    
    params = chose_rand_params(gen_params)
    watermark = create_watermark(params)
    
    # apply watermark
    watermarked_img = apply_watermark(image, watermark, params)
    # plot
    plt.imshow(cv2.cvtColor(watermarked_img, cv2.COLOR_BGR2RGB))
    # flatten and save image
    watermarked_img = cv2.cvtColor(watermarked_img, cv2.COLOR_BGRA2BGR)
    figName = os.path.basename(input_folder)
    output_name = output_folder + '\\' + figName
    cv2.imwrite(output_name, watermarked_img)