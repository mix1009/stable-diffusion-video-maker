# Stable Diffusion Video Maker (sdvm)

sdvm is a video generation tool for Stable Diffusion.
The code is based on CrossAttentionControl which supports prompt editing.

## Features
* animate between seeds, prompts, and prompt edits.
* can assign different frames between keyframes.
* keyframes can be animated using different curves (linear, bezier, bezier2...)
* upsampling using ESRGAN

## Dependencies:
* python libraries: 
`torch transformers diffusers==0.4.1 numpy PIL tqdm difflib librosa realesrgan`
* executable: `ffmpeg`

## Google Colab notebook:
* https://colab.research.google.com/drive/15x55IoGOZHqozTPKDlX7ZBG0duTV-lj4?usp=sharing

## Credits:
CrossAttentionControl : https://github.com/bloc97/CrossAttentionControl

Upsampling code : https://github.com/nateraw/stable-diffusion-videos
