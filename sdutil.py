import os
import torch
import numpy as np
from PIL import Image
import math
import cv2
import subprocess

    
def encode(srcpath, video_file, fps=10, imgfilepattern='frame%05d.png', audiofile=None, preview=False, print_cmd=False):
    cmd = ['ffmpeg', '-y']
    if audiofile is not None:
        cmd.extend(['-i', audiofile])
    cmd.extend(['-vcodec', 'png', '-r', str(fps), '-start_number', str(0)])
    cmd.extend(['-i', f'{srcpath}/{imgfilepattern}'])
    if preview:
        cmd.extend(['-frames:v', str(200)])
    cmd.extend([
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '17',
        '-preset', 'veryfast',
        '-pattern_type', 'sequence',
         video_file
    ])
    if print_cmd:
        print(' '.join(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)


def show_video(video_path, width=None, height=None):
    from IPython.display import display, HTML
    from base64 import b64encode

    if not os.path.exists(video_path):
        raise RuntimeError('video file not found')
        return
    
    videosize = "width=512"
    if width is not None:
        videosize = f"width={width} "
    elif height is not None:
        videosize = f"height={height}"

    mp4 = open(video_path, 'rb').read()
    data_url = 'data:simul2/mp4;base64,' + b64encode(mp4).decode()
    return HTML("""
        <video {} controls>
              <source src="{}" type="video/mp4">
        </video>
        """.format(videosize, data_url))

def upsample_images(srcpath, dstpath, fileprefix='frame'):
    from tqdm.auto import tqdm
    from PIL import Image
    from upsampling import RealESRGANModel
    import glob

    os.makedirs(dstpath, exist_ok=True)
    upsample_pipeline = RealESRGANModel.from_pretrained('nateraw/real-esrgan')

    def np2PIL(img_np):
        return Image.fromarray(np.uint8(img_np * 255))
    def PIL2np(image):
        return np.float32(np.asarray(image)) / 255.0

    def upsample_image(image):
        np_image = PIL2np(image)
        return upsample_pipeline.forward(np_image)
    
    files = glob.glob(os.path.join(srcpath , f'{fileprefix}*.png'))
    for file in tqdm(files, desc="upsample"):
        src = os.path.normpath(file)
        dst = os.path.join(dstpath, src.split(os.sep)[-1])
#         print(f'{src} -> {dst}')
        if os.path.exists(dst) and os.stat(dst).st_mtime > os.stat(src).st_mtime:
            continue
        image = Image.open(src)
        image = upsample_image(image)
        tmp = os.path.join(dstpath, '_tmp_'+src.split(os.sep)[-1])
        try:
            image.save(tmp)
            if os.path.exists(dst):
                os.unlink(dst)
            os.rename(tmp, dst)
        except:
            image.save(dst)
                
    upsample_pipeline = None
    torch.cuda.empty_cache()
    import gc
    gc.collect()

def image_grid(imgs, cols=3):
    def _grid(imgs, rows, cols):
        
        w, h = imgs[0].size
        grid = Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size

        for i, img in enumerate(imgs):
            grid.paste(img, box=(i%cols*w, i//cols*h))
        return grid
    rows = math.ceil(len(imgs) / cols)
    if len(imgs)<cols:
        cols = len(imgs)
    
    return _grid(imgs, rows, cols)

import scipy.interpolate
def interp(arr, new_size):
    arr1_interp = scipy.interpolate.interp1d(np.arange(arr.size),arr)
    return arr1_interp(np.linspace(0,arr.size-1, new_size))


def opencvtransform(pil_img, angle, translation_x, translation_y, zoom, wrap):
    numpy_image = np.array(pil_img)  
    prev_img_cv2 = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 
    
    center = (pil_img.size[0] // 2, pil_img.size[1] // 2)
    trans_mat = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
    rot_mat = cv2.getRotationMatrix2D(center, angle, zoom)
    trans_mat = np.vstack([trans_mat, [0,0,1]])
    rot_mat = np.vstack([rot_mat, [0,0,1]])
    xform = np.matmul(rot_mat, trans_mat)

    opencv_image = cv2.warpPerspective(
        prev_img_cv2,
        xform,
        (prev_img_cv2.shape[1], prev_img_cv2.shape[0]),
        borderMode=cv2.BORDER_WRAP if wrap else cv2.BORDER_REPLICATE
        )
    
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(color_coverted)


def color_pop(pil_img, saturation=0.0, hue=0.0, lower=[0,140,120], upper=[255,255,255]):
    numpy_image = np.array(pil_img)  
    img = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) 

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:,:,0]
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    snew = cv2.add(s, saturation)
    hnew = cv2.add(h, hue)
    hsvnew = cv2.merge([hnew,snew,v])
    gray = cv2.cvtColor(hsvnew, cv2.COLOR_HSV2BGR)

    #lower_green = np.array([50,100,50])
    #upper_green = np.array([70,255,255])


    #lower_red = np.array([160,100,50])
    #upper_red = np.array([180,255,255])


    #set the bounds for the red hue
    lower_red = np.array(lower)
    upper_red = np.array(upper)

    #create a mask using the bounds set
    mask = cv2.inRange(hsv, lower_red, upper_red)
    #create an inverse of the mask
    mask_inv = cv2.bitwise_not(mask)
    #Filter only the red colour from the original image using the mask(foreground)
    res = cv2.bitwise_and(img, img, mask=mask)
    background = cv2.bitwise_and(gray, gray, mask = mask_inv)
    added_img = cv2.add(res, background)

    color_coverted = cv2.cvtColor(added_img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(color_coverted)
