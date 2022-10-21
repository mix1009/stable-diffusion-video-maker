# Stable Diffusion Video Maker (sdvm)

sdvm is a video generation tool for Stable Diffusion.
The code is based on CrossAttentionControl which supports prompt editing.

## Features
* animate between seeds, prompts, and prompt edit weights.
* can insert different number of frames between keyframes.
* keyframes can be animated using different curves (linear, bezier, bezier2...)
* supports negative prompts
* cache keyframes
* upsampling using RealESRGAN
* encode video using ffmpeg

## Dependencies:
* python libraries: 
`torch transformers diffusers==0.4.1 numpy PIL tqdm difflib librosa realesrgan`
* executable: `ffmpeg`

## Example code:
```
import sdvm

v = sdvm.VideoMaker('project_name',
                    prompt='dog with hat', 
                    num_frames=10,
                    fps=10,
                    increase_mode=sdvm.IncreaseMode.linear)

v.add(edit_weights=[('hat', -0.5)], seed=1001)
v.add(edit_weights=[('hat', 1.5)], seed=1001)
v.add(prompt='cat', negative_prompt='yellow', seed=1002, num_frames=15)

# creates images in sdout folder
v.make()

# encode video file and display in python notebook
v.encode(show=True)
```

## Project Structure:
 * projects/project_name is the root of the project. (projects can be changed by passing basepath in VideoMaker)
 * projects/project_name/sdout : SD images are stored here.
 * projects/project_name/project_name.mp4 : video file path
 * projects/project_name/upsample : upsampled images.
* projects/project_name/project_name_upsample.mp4 : upsampled video file path
 
## Google Colab notebook:
* https://colab.research.google.com/drive/15x55IoGOZHqozTPKDlX7ZBG0duTV-lj4?usp=sharing

## Credits:
CrossAttentionControl : https://github.com/bloc97/CrossAttentionControl

Upsampling code : https://github.com/nateraw/stable-diffusion-videos
