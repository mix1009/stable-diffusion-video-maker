from crossattentioncontrol import sd, get_latent, get_text_embedding, lerp_latent_and_embedding
import sdutil
from tqdm.auto import tqdm
import numpy as np
import os, glob, shutil
import torch

class KeyFrame:        
    def __init__(self, video_project,
                 prompt=None,
                 prompt_edit=None,
                 edit_weights=[],
                 negative_prompt=None,
                 seed=None,
                 strength=1.0,
                 num_frames=30,
                 start_frame=0,
                 overwrite=None):
        self.video_project = video_project
        self.prompt = prompt
        self.prompt_edit = prompt_edit
        self.edit_weights = edit_weights
        self.negative_prompt = negative_prompt
        self.seed = seed
        self.strength = strength
        self.width = video_project.width
        self.height = video_project.height
        self.num_frames = num_frames
        self.start_frame = start_frame
        self.start_sec = float(start_frame / video_project.fps)
        self.overwrite = overwrite
        self.hash = self.calc_hash()

    def cache_path(self):
        v = self.video_project
        return f'{v.cache_path}/{self.hash}_{v.width}x{v.height}.png'
    
    def calc_hash(self):
        import hashlib
        v = self.video_project
        t = f'n_{self.prompt}_{self.seed}_{self.prompt_edit}_{self.negative_prompt}'
        t += f'_{self.edit_weights}_{self.strength}_{v.num_inference_steps}'
        t += f'_{v.width}_{v.height}_{v.guidance_scale}'
        return hashlib.md5(t.encode('utf-8')).hexdigest()

    def __str__(self):
        prompt = self.prompt
        if len(prompt)>60:
            prompt = prompt[:60]+'...'
        txt = f'{self.start_frame:4} {self.start_sec:3.1f}: {prompt} ({self.seed})'
        if len(self.edit_weights)>0:
            w = f'{self.edit_weights}'
            if len(w)>50:
                w = w[:50] + '...'
            txt += f'\n\t\t{w}'
        return txt
    
    def image(self):
        import os
        from PIL import Image
        v = self.video_project
        path = self.cache_path()
        if os.path.exists(path):
            return Image.open(path)
        img = v.sd(prompt=self.prompt,
                   prompt_edit=self.prompt_edit,
                   edit_weights=self.edit_weights,
                   seed=self.seed)
        img.save(path)
        return img
    
    def img2img(self):
        v = self.video_project
        img = v.sd(prompt=self.prompt,
                   prompt_edit=self.prompt_edit,
                   edit_weights=self.edit_weights,
                   seed=self.seed)
        return img

    def to_dict(self):
        m = {}
        m['prompt'] = self.prompt
        m['prompt_edit'] = self.prompt_edit
        m['negative_prompt'] = self.negative_prompt
        m['seed'] = self.seed
        m['edit_weights'] = self.edit_weights
        m['strength'] = self.strength
        m['num_frames'] = self.num_frames
        m['start_frame'] = self.start_frame
        return m

class IncreaseMode:
    
    def linear(t):
        return t
    def bezier(t):
        return t * t * (3.0 - 2.0 * t)
    def bezier2(t):
        return IncreaseMode.bezier(IncreaseMode.bezier(t))
    def bezier3(t):
        return IncreaseMode.bezier(IncreaseMode.bezier2(t))

    def fix(t, v=0.005):
        if t < v: return 0.0
        if t > (1-v): return 1.0
        return t
    def bezierf(t):
        return IncreaseMode.fix(IncreaseMode.bezier(t))
    def bezier2f(t):
        return IncreaseMode.fix(IncreaseMode.bezier2(t))
    def bezier3f(t):
        return IncreaseMode.fix(IncreaseMode.bezier3(t))

class VideoMaker:
    def __init__(self, name,
                 prompt="",
                 prompt_edit=None,
                 negative_prompt="",
                 width=512, height=512,
                 fps=10,
                 num_frames=90,
                 num_inference_steps=35,
                 guidance_scale=7.5,
                 init_image_strength=0.5,
                 basepath='projects',
                 increase_mode=IncreaseMode.linear,
                 load_json=False):
        self.name = name
        self.basepath = basepath
        self.projectpath = f'{self.basepath}/{self.name}'
        self.prompt = prompt
        self.prompt_edit = prompt_edit
        self.width = width
        self.height = height
        self.fps = fps
        self.audiofile = None
        self.audiopeak = None
        self.animation_dict = {}
        self.num_frames = num_frames
        self.num_inference_steps = num_inference_steps
        self.init_image = None
        self.init_image_strength = init_image_strength
        self.guidance_scale = guidance_scale
        self.increase_mode = increase_mode
        self.negative_prompt = negative_prompt

        self.key_frames = []
        self.total_frame_count = 0
        self.initialize_path()
        
        if load_json:
            self.load_json()
        
    def initialize_path(self):
        import os
        self.projectpath = f'{self.basepath}/{self.name}'
        self.cache_path = f'{self.projectpath}/cache'
        
        self.sdout_path = f'{self.projectpath}/sdout'
        self.upsample_path = f'{self.projectpath}/upsample'
        self.processed_path = f'{self.projectpath}/processed'
        self.processed_upsample_path = f'{self.projectpath}/processed_upsample'
        
        os.makedirs(self.projectpath, exist_ok=True)
        os.makedirs(self.sdout_path, exist_ok=True)
        os.makedirs(self.cache_path, exist_ok=True)
        
    def load_audiofile(self, audiofile):
        import audiopeak
        self.audiofile = audiofile
        self.audiopeak = audiopeak.AudioPeak(audiofile, fps=self.fps)
        self.total_audio_frames = len(self.audiopeak.strength)
        return self.audiopeak
        
    def sd(self, prompt=None, prompt_edit=None, edit_weights=[], negative_prompt=None,
           seed=None, latent=None, text_embedding=None):
        if prompt is None:
            prompt = self.prompt
        if prompt_edit is None:
            prompt_edit = self.prompt_edit
        if negative_prompt is None:
            negative_prompt = self.negative_prompt
        return sd(prompt=prompt,
                  seed=seed,
                  latent=latent,
                  text_embedding=text_embedding,
                  prompt_edit=prompt_edit,
                  edit_weights=edit_weights,  
                  negative_prompt=negative_prompt,
                  width=self.width, height=self.height,
                  steps=self.num_inference_steps,
                  guidance_scale=self.guidance_scale,
                  init_image=self.init_image,
                  init_image_strength=self.init_image_strength,
                 )
    
    def generate(self, seed, count=6, prompt=None, prompt_edit=None, negative_prompt=None, edit_weights=[], cols=3):
        imgs = []
        if prompt is None:
            prompt = self.prompt
        if prompt_edit is None:
            prompt_edit = self.prompt_edit
        if negative_prompt is None:
            negative_prompt = self.negative_prompt
        images = []
        for idx in tqdm(range(count)):
            k = KeyFrame(self,
                         seed=seed+idx,
                         prompt=prompt,
                         prompt_edit=prompt_edit,
                         edit_weights=edit_weights,
                         negative_prompt=negative_prompt,
                         num_frames=1,
                         start_frame=0,
                         strength=1.0)
            images.append(k.image())
            if len(images) == 3:
                display(sdutil.image_grid(images, cols=3))
                images = []
                    
        if len(images) > 0:
            display(sdutil.image_grid(images, cols=3))
    
    def add(self, seed=None, prompt=None, prompt_edit=None, edit_weights=[], negative_prompt=None, num_frames=None, overwrite=None):
        if prompt is None:
            prompt = self.prompt
        if prompt_edit is None:
            prompt_edit = self.prompt_edit
        if negative_prompt is None:
            negative_prompt = self.negative_prompt
        if num_frames is None:
            num_frames = self.num_frames
        if len(self.key_frames)==0:
            num_frames = 1

        self.total_frame_count += num_frames

        k = KeyFrame(self,
                     seed=seed,
                     prompt=prompt,
                     prompt_edit=prompt_edit,
                     edit_weights=edit_weights,
                     negative_prompt=negative_prompt,
                     num_frames=num_frames,
                     start_frame=self.total_frame_count,
                     strength=1.0,
                     overwrite=overwrite)
        
        self.key_frames.append(k)
        
    def _get_increase_function(self, increase_mode):
        if callable(increase_mode):
            return self.increase_mode
        else:
            for fname in dir(IncreaseMode):
                if fname.startswith('_'): continue
                if fname == increase_mode:
                    return IncreaseMode.__dict__[fname]
        raise RunTimeError(f'increase_mode {increase_mode} failed to resolve')

    def make(self, overwrite=False, show_frame=False):
        def w_expand(w):
            r = []
            for kl,v in w:
                if isinstance(kl, list) or isinstance(kl, tuple):
                    pass
                else:
                    kl = [kl]
                for k in kl:
                    r.append((k,v))
            return r
        def w_lerp(w1, w2, t):
            w1v = {}
            for k,v in w2:
                w1v[k] = v # in case key is not in w1
            for k,v in w1:
                w1v[k] = v
            return [(k, w1v[k]*(1-t) + v2*t) for k,v2 in w2]
        
        from PIL import ImageDraw 
        
        keyframes = self.key_frames[:]
        k1, *keyframes = keyframes
        
        self.save_json()
        
        keynum = 0
        framenum = 0
        
        mod_fn = self._get_increase_function(self.increase_mode)

        with tqdm(total=self.total_frame_count) as pbar:
            while len(keyframes)>0:
                k2, *keyframes = keyframes
                
                if k2.overwrite is not None:
                    overwrite=k2.overwrite

                seed=k2.seed

                latent1 = get_latent(k1.seed, self.width, self.height)
                latent2 = get_latent(k2.seed, self.width, self.height)
                    
                embed1 = get_text_embedding(k1.prompt)
                embed2 = get_text_embedding(k2.prompt)
                    
                w1 = w_expand(k1.edit_weights)
                w2 = w_expand(k2.edit_weights)
                
                last_t = None
                last_img = None
                img = None

                for t in np.linspace(0,1,k2.num_frames+1):
                    if t == 1 and len(keyframes)>0: break

                    t = mod_fn(t)
    
                    latent, embed = lerp_latent_and_embedding(latent1, latent2, embed1, embed2, t)
                    w = w_lerp(w1, w2, t)


                    savepath = f'{self.sdout_path}/frame{framenum:05}.png'
                    if overwrite or not os.path.exists(savepath):
                        pbar.set_description(f'frame={framenum}')
                        pbar.refresh()
                        if t != last_t or last_img is None:
                            img = self.sd(seed=seed,
                                      latent=latent,
                                      text_embedding=embed,
                                      prompt=k2.prompt,
                                      prompt_edit=k2.prompt_edit,
                                      negative_prompt=k2.negative_prompt,
                                      edit_weights=w)
                        else:
#                             print(f'reuse frame {framenum} t={t}')
                            img = last_img

                        img.save(savepath)
                    pbar.update(1)
                    last_t = t
                    last_img = img
                    framenum+=1
                k1 = k2
                keynum += 1
            
    def print_info(self):
        for k in self.key_frames:
            print(k,k.hash)
            
    def show(self, show_image=False):
        images = []
        for k in self.key_frames:
            print(k)
            if show_image:
                images.append(k.image())
                if len(images) == 3:
                    display(sdutil.image_grid(images, cols=3))
                    images = []
                    
        if len(images) > 0:
            display(sdutil.image_grid(images, cols=3))

    def preview(self, one_image=False):
        images = []
        last_hash = None
        for k in self.key_frames:
            if last_hash == k.hash: continue            
            images.append(k.image())
            if len(images) == 2 and not one_image:
                display(sdutil.image_grid(images, cols=2))
                images = []
            last_hash = k.hash
                    
                    
        if len(images) > 0:
            display(sdutil.image_grid(images, cols=2))


    def encode(self, show=False):
        mp4 = f'{self.projectpath}/{self.name}.mp4'
        sdutil.encode(self.sdout_path, mp4, fps=self.fps)
        if show:
            return sdutil.show_video(mp4)
        else:
            return mp4
        
    def upsample(self):        
        src = self.sdout_path
        dst = self.upsample_path
        mp4 = f'{self.projectpath}/{self.name}_upsample.mp4'
        
        if len(glob.glob(f'{self.processed_path}/*.png')) > 10:
            print('upsample processed!')
            src = self.processed_path
            dst = self.processed_upsample_path
            mp4 = f'{self.projectpath}/{self.name}_processed_upsample.mp4'
            
        os.makedirs(dst, exist_ok=True)

        sdutil.upsample_images(src, dst)
        sdutil.encode(dst, mp4)
        
    def animate(self, animation_dict):
        from PIL import Image
        anim = animation_dict
        self.animation_dict = anim
        src = self.sdout_path
        dst = self.processed_path
        os.makedirs(dst, exist_ok=True)
        
        angle = 0.0
        zoom = 1.0
        translation_x = 0.0
        translation_y = 0.0
        color_pop_contrast = None
        color_pop_lower = [0,0,180]
        color_pop_upper = [255,255,255]
        for idx, f in enumerate(tqdm(glob.glob(f'{self.sdout_path}/frame*.png'), desc='animate')):
            filename = os.path.basename(f)
            try:
                angle = anim['angle'][idx]
            except: pass
            try:
                zoom = anim['zoom'][idx]
            except: pass
            try:
                translation_x = anim['translation_x'][idx]
            except: pass
            try:
                translation_y = anim['translation_y'][idx]
            except: pass
            try:
                color_pop_contrast = anim['color_pop_contrast'][idx]
            except: pass
            try:
                lower = anim['color_pop_lower'][idx]
                if len(lower)==3:
                    print(f'lower {lower}')
                    color_pop_lower = lower
            except: pass
            try:
                upper = anim['color_pop_upper'][idx]
                if len(upper)==3:
                    print(f'upper {upper}')
                    color_pop_upper = upper
            except: pass

#             print(f'{filename}: angle={angle:.2f} zoom={zoom:.2f}')
            img = Image.open(f)
            img = sdutil.opencvtransform(img, angle, translation_x, translation_y, zoom, False)
            if color_pop_contrast is not None:
                img = sdutil.color_pop(img, color_pop_contrast, lower=color_pop_lower, upper=color_pop_upper)

            img.save(f'{self.processed_path}/{filename}')
          
        mp4 = f'{self.projectpath}/{self.name}_processed.mp4'
        sdutil.encode(dst, mp4)


    def to_json(self):
        import json

        m = {}
        m['name'] = self.name
        m['prompt'] = self.prompt
        m['prompt_edit'] = self.prompt_edit
        m['negative_prompt'] = self.negative_prompt
        m['width'] = self.width
        m['height'] = self.height
        m['basepath'] = self.basepath
        m['fps'] = self.fps
        m['audiofile'] = self.audiofile
        m['num_frames'] = self.num_frames
        m['num_inference_steps'] = self.num_inference_steps
        m['animation_dict'] = self.animation_dict
        
        m['key_frames'] = [k.to_dict() for k in self.key_frames]
        if callable(self.increase_mode):
            m['increase_mode'] = self.increase_mode.__name__
        else:
            m['increase_mode'] = self.increase_mode
        
        return json.dumps(m, indent=2, sort_keys=False)
    
    def save_json(self, path=None):
        if path==None:
            path = f'{self.projectpath}/{self.name}.json'
        open(path, 'w').write(self.to_json())
    
    def load_json(self, path=None):
        if path==None:
            path = f'{self.projectpath}/{self.name}.json'
        if not os.path.exists(path):
            print(f'load_json: {path} does not exist.')
            return
        import json
        m = json.load(open(path))
        self.name = m['name']
        self.prompt = m['prompt']
        self.prompt_edit = m['prompt_edit']
        if 'negative_prompt' in m.keys():
            self.negative_prompt = m['negative_prompt']
        else:
            self.negative_prompt = ""
        self.width = m['width']
        self.height = m['height']
        self.basepath = m['basepath']
        self.fps = m['fps']
        self.audiofile = m['audiofile']
        self.num_frames = m['num_frames']
        self.num_inference_steps = m['num_inference_steps']
        self.animation_dict = m['animation_dict']
        self.increase_mode = m['increase_mode'] if 'increase_mode' in m.keys() else None

        self.key_frames = []
        
        for d in m['key_frames']:
            negp = ""
            if 'negative_prompt' in d.keys():
                negp = d['negative_prompt']
            k = KeyFrame(self,
                         prompt=d['prompt'],
                         prompt_edit=d['prompt_edit'],
                         edit_weights=[(k,v) for k,v in d['edit_weights']],
                         negative_prompt=negp,
                         seed=d['seed'],
                         strength=d['strength'],
                         num_frames=d['num_frames'],
                         start_frame=d['start_frame'])
            self.add_keyframe(k)
            self.total_frame_count += k.num_frames
            
        return m
    
    def delete_cache(self):
        shutil.rmtree(self.cache_path)
        os.makedirs(self.cache_path, exist_ok=True)
    
    def clean(self):
        shutil.rmtree(self.sdout_path)
        os.makedirs(self.sdout_path, exist_ok=True)

    def archive(self):
        import zipfile
        import glob
        with zipfile.ZipFile(f'{self.basepath}/{self.name}.zip', 'w') as myzip:
            myzip.write(f'{self.basepath}/{self.name}/{self.name}.json')
            for f in glob.glob(f'{self.basepath}/{self.name}/sdout/*.png'):
                myzip.write(f)
            for f in glob.glob(f'{self.basepath}/{self.name}/cache/*.png'):
                myzip.write(f)
            myzip.close()

    def retire(self):
        self.save_json()
        src = f'{self.basepath}/{self.name}/{self.name}.json'
        dst = f'{self.basepath}/#retired/{self.name}.json'
        os.makedirs(f'{self.basepath}/#retired', exist_ok=True)
        shutil.copyfile(src, dst)
        self.archive()
        self.rmtree()

    def rmtree(self):
        path = f'{self.basepath}/{self.name}'
        shutil.rmtree(path)
        print(f'deleted {path}')
    
    def make_audio_preview(self):
        from PIL import Image, ImageDraw, ImageFont
        import os, shutil
        peak = self.audiopeak
        font = ImageFont.truetype("font/Roboto-Regular.ttf", 16)

        path = f'{self.projectpath}/audio_preview'
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

        name = os.path.basename(self.audiofile)
        nameonly = name[:name.rfind('.')]

        def prepend_zeros(arr, count=10):
            return np.concatenate((np.zeros(count), arr))
        strength = prepend_zeros(peak.strength)
        beats = prepend_zeros(peak.beats)
        perc = prepend_zeros(peak.perc)
        harm = prepend_zeros(peak.harm)    

        def draw_graph(arr, frame, y, title, color, beats=None):
            for idx, val in enumerate(arr[frame:]):
                x = idx*7
                if x > self.width: break
                yval = val*50
                if beats is not None and beats[frame+idx]==1:                
                    draw.rectangle([(x, y+20), (x+5, y+20-yval)], fill='#ffffc0')
                else:
                    draw.rectangle([(x, y+20), (x+5, y+20-yval)], fill=color)
            draw.text((80, y-50), title, color, font=font)

        for frame in tqdm(range(0, self.total_audio_frames)):

            img = Image.new('RGB', (self.width, self.height), color = 'black')
            draw = ImageDraw.Draw(img)
            draw.rectangle([(71, 80), (72, 480)], fill='#404040')

            draw_graph(strength, frame, 150, 'strength', '#ff0000', beats)
            draw_graph(beats,    frame, 250, 'beats', '#ffff80')
            draw_graph(perc,     frame, 350, 'perc', '#80ff80', beats)
            draw_graph(harm,     frame, 450, 'harm', '#8080ff', beats)

            draw.text((10, 10),f'[{frame}/{self.total_audio_frames-1}] {name}',(255,255,255),font=font)

            txt = f'{self.fps} fps / {self.audiopeak.duration:.1f}s'
            text_w = draw.textlength(txt, font=font)
            draw.text((self.width-text_w-10, 10), txt,'#808080',font=font)

            img.save(f'{path}/frame{frame:05}.png')

        sdutil.encode(path, f'{self.projectpath}/{nameonly}_{self.fps}fps.mp4', audiofile=self.audiofile, fps=self.fps)    