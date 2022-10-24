# code based on
# https://github.com/bloc97/CrossAttentionControl/blob/main/CrossAttention_Release_NoImages.ipynb
 
import torch
from transformers import CLIPModel, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel

_model = None

class SDModel:
    def __init__(self, model_path_diffusion, model_path_clip, auth_token=None):
        self.model_path_diffusion = model_path_diffusion
        self.model_path_clip = model_path_clip
        self.auth_token = auth_token
        
        # huggingface auth token
        if self.auth_token is None:
            self.auth_token = True
        

        self.init_model()
        
    def init_model(self):
        print("Loading models...")

        self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.model_path_clip)
        self.clip_model = CLIPModel.from_pretrained(self.model_path_clip, torch_dtype=torch.float16)
        self.clip = self.clip_model.text_model

        self.unet = UNet2DConditionModel.from_pretrained(self.model_path_diffusion,
                                                         subfolder="unet", 
                                                         use_auth_token=self.auth_token,
                                                         revision="fp16",
                                                         torch_dtype=torch.float16)
        self.vae = AutoencoderKL.from_pretrained(self.model_path_diffusion,
                                                 subfolder="vae",
                                                 use_auth_token=self.auth_token,
                                                 revision="fp16",
                                                 torch_dtype=torch.float16)

        self.device = "cuda"
        self.unet.to(self.device)
        self.vae.to(self.device)
        self.clip.to(self.device)
        print("Loaded all models")

def init_model(model_path_diffusion = "CompVis/stable-diffusion-v1-4",
               model_path_clip = "openai/clip-vit-large-patch14",
               auth_token = None):
    global _model
    _model = SDModel(model_path_diffusion, model_path_clip, auth_token)
    return _model

def get_model():
    global _model
    if _model is None:
        _model = init_model()
    return _model

import numpy as np
import random
from PIL import Image
from diffusers import LMSDiscreteScheduler
from tqdm.auto import tqdm
from torch import autocast
from difflib import SequenceMatcher

def init_attention_weights(weight_tuples):
    clip_tokenizer = get_model().clip_tokenizer
    tokens_length = clip_tokenizer.model_max_length
    weights = torch.ones(tokens_length)
    unet = get_model().unet
    device = get_model().device
    
    for i, w in weight_tuples:
        if i < tokens_length and i >= 0:
            weights[i] = w
    
    
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_weights = weights.to(device)
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_weights = None
    

def init_attention_edit(tokens, tokens_edit):
    clip_tokenizer = get_model().clip_tokenizer
    unet = get_model().unet
    device = get_model().device
    
    tokens_length = clip_tokenizer.model_max_length
    mask = torch.zeros(tokens_length)
    indices_target = torch.arange(tokens_length, dtype=torch.long)
    indices = torch.zeros(tokens_length, dtype=torch.long)

    tokens = tokens.input_ids.numpy()[0]
    tokens_edit = tokens_edit.input_ids.numpy()[0]
    
    for name, a0, a1, b0, b1 in SequenceMatcher(None, tokens, tokens_edit).get_opcodes():
        if b0 < tokens_length:
            if name == "equal" or (name == "replace" and a1-a0 == b1-b0):
                mask[b0:b1] = 1
                indices[b0:b1] = indices_target[a0:a1]

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.last_attn_slice_mask = mask.to(device)
            module.last_attn_slice_indices = indices.to(device)
        if module_name == "CrossAttention" and "attn1" in name:
            module.last_attn_slice_mask = None
            module.last_attn_slice_indices = None


def init_attention_func():
    #ORIGINAL SOURCE CODE: https://github.com/huggingface/diffusers/blob/91ddd2a25b848df0fa1262d4f1cd98c7ccb87750/src/diffusers/models/attention.py#L276
    
    unet = get_model().unet

    def new_attention(self, query, key, value):
        # TODO: use baddbmm for better performance
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attn_slice = attention_scores.softmax(dim=-1)
        # compute attention output
        
        if self.use_last_attn_slice:
            if self.last_attn_slice_mask is not None:
                new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
            else:
                attn_slice = self.last_attn_slice

            self.use_last_attn_slice = False

        if self.save_last_attn_slice:
            self.last_attn_slice = attn_slice
            self.save_last_attn_slice = False

        if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
            attn_slice = attn_slice * self.last_attn_slice_weights
            self.use_last_attn_weights = False
        
        hidden_states = torch.matmul(attn_slice, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states
    
    def new_sliced_attention(self, query, key, value, sequence_length, dim):
        
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                torch.matmul(query[start_idx:end_idx], key[start_idx:end_idx].transpose(1, 2)) * self.scale
            )  # TODO: use baddbmm for better performance
            attn_slice = attn_slice.softmax(dim=-1)
            
            if self.use_last_attn_slice:
                if self.last_attn_slice_mask is not None:
                    new_attn_slice = torch.index_select(self.last_attn_slice, -1, self.last_attn_slice_indices)
                    attn_slice = attn_slice * (1 - self.last_attn_slice_mask) + new_attn_slice * self.last_attn_slice_mask
                else:
                    attn_slice = self.last_attn_slice
                
                self.use_last_attn_slice = False
                    
            if self.save_last_attn_slice:
                self.last_attn_slice = attn_slice
                self.save_last_attn_slice = False
                
            if self.use_last_attn_weights and self.last_attn_slice_weights is not None:
                attn_slice = attn_slice * self.last_attn_slice_weights
                self.use_last_attn_weights = False
            
            attn_slice = torch.matmul(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention":
            module.last_attn_slice = None
            module.use_last_attn_slice = False
            module.use_last_attn_weights = False
            module.save_last_attn_slice = False
            module._sliced_attention = new_sliced_attention.__get__(module, type(module))
            module._attention = new_attention.__get__(module, type(module))
            
def use_last_tokens_attention(use=True):
    unet = get_model().unet
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_slice = use
            
def use_last_tokens_attention_weights(use=True):
    unet = get_model().unet
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.use_last_attn_weights = use
            
def use_last_self_attention(use=True):
    unet = get_model().unet
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.use_last_attn_slice = use
            
def save_last_tokens_attention(save=True):
    unet = get_model().unet
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn2" in name:
            module.save_last_attn_slice = save
            
def save_last_self_attention(save=True):
    unet = get_model().unet
    for name, module in unet.named_modules():
        module_name = type(module).__name__
        if module_name == "CrossAttention" and "attn1" in name:
            module.save_last_attn_slice = save

@torch.no_grad()
def stablediffusion(prompt="",
                    negative_prompt="",
                    text_embedding=None,
                    prompt_edit=None,
                    prompt_edit_token_weights=[],
                    prompt_edit_tokens_start=0.0,
                    prompt_edit_tokens_end=1.0,
                    prompt_edit_spatial_start=0.0,
                    prompt_edit_spatial_end=1.0,
                    guidance_scale=7.5,
                    steps=50,
                    seed=None,
                    init_latents=None,
                    width=512,
                    height=512,
                    init_image=None,
                    init_image_strength=0.5,
                    disable_tqdm=False,
                   ):
    clip_tokenizer = get_model().clip_tokenizer
    clip = get_model().clip
    unet = get_model().unet
    vae = get_model().vae
    device = get_model().device

    #Change size to multiple of 64 to prevent size mismatches inside model
    width = width - width % 64
    height = height - height % 64
    
    #If seed is None, randomly select seed from 0 to 2^32-1
    if seed is None: seed = random.randrange(2**32 - 1)
    generator = torch.cuda.manual_seed(seed)
    
    #Set inference timesteps to scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
    scheduler.set_timesteps(steps)
    
    #Preprocess image if it exists (img2img)
    if init_image is not None:
        #Resize and transpose for numpy b h w c -> torch b c h w
        init_image = init_image.resize((width, height), resample=Image.Resampling.LANCZOS)
        init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
        init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))
        
        #If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
        if init_image.shape[1] > 3:
            init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])
            
        #Move image to GPU
        init_image = init_image.to(device)
        
        #Encode image
        with autocast(device):
            init_latent = vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215
            
        t_start = steps - int(steps * init_image_strength)
            
    else:
        init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
        t_start = 0
    
    #Generate random normal noise
    noise = torch.randn(init_latent.shape, generator=generator, device=device)
    
    if init_latents is not None:
        noise = init_latents
    #latent = noise * scheduler.init_noise_sigma
    latent = scheduler.add_noise(init_latent, noise, torch.tensor([scheduler.timesteps[t_start]], device=device)).to(device)
    
    #Process clip
    with autocast(device):
        tokens_unconditional = clip_tokenizer(negative_prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_unconditional = clip(tokens_unconditional.input_ids.to(device)).last_hidden_state

        tokens_conditional = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state
        if text_embedding is not None:
            embedding_conditional = text_embedding

        #Process prompt editing
        if prompt_edit is not None:
            tokens_conditional_edit = clip_tokenizer(prompt_edit, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
            embedding_conditional_edit = clip(tokens_conditional_edit.input_ids.to(device)).last_hidden_state
            
            init_attention_edit(tokens_conditional, tokens_conditional_edit)
            
        init_attention_func()
        init_attention_weights(prompt_edit_token_weights)
            
        timesteps = scheduler.timesteps[t_start:]
        
        for i, t in tqdm(enumerate(timesteps), total=len(timesteps), disable=disable_tqdm):
            t_index = t_start + i

            #sigma = scheduler.sigmas[t_index]
            latent_model_input = latent
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            #Predict the unconditional noise residual
            noise_pred_uncond = unet(latent_model_input, t, encoder_hidden_states=embedding_unconditional).sample
            
            #Prepare the Cross-Attention layers
            if prompt_edit is not None:
                save_last_tokens_attention()
                save_last_self_attention()
            else:
                #Use weights on non-edited prompt when edit is None
                use_last_tokens_attention_weights()
                
            #Predict the conditional noise residual and save the cross-attention layer activations
            noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_conditional).sample
            
            #Edit the Cross-Attention layer activations
            if prompt_edit is not None:
                t_scale = t / scheduler.num_train_timesteps
                if t_scale >= prompt_edit_tokens_start and t_scale <= prompt_edit_tokens_end:
                    use_last_tokens_attention()
                if t_scale >= prompt_edit_spatial_start and t_scale <= prompt_edit_spatial_end:
                    use_last_self_attention()
                    
                #Use weights on edited prompt
                use_last_tokens_attention_weights()

                #Predict the edited conditional noise residual using the cross-attention masks
                noise_pred_cond = unet(latent_model_input, t, encoder_hidden_states=embedding_conditional_edit).sample
                
            #Perform guidance
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            latent = scheduler.step(noise_pred, t_index, latent).prev_sample

        #scale and decode the image latents with vae
        latent = latent / 0.18215
        image = vae.decode(latent.to(vae.dtype)).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image[0] * 255).round().astype("uint8")
    return Image.fromarray(image)

def prompt_token(prompt, index):
    tokens = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True).input_ids[0]
    return clip_tokenizer.decode(tokens[index:index+1])


###############

@torch.no_grad()
def get_latent(seed, width=512, height=512):
    device = get_model().device
    unet = get_model().unet

    width = width - width % 64
    height = height - height % 64
    
    generator = torch.cuda.manual_seed(seed)

    init_latent = torch.zeros((1, unet.in_channels, height // 8, width // 8), device=device)
    noise = torch.randn(init_latent.shape, generator=generator, device=device)
    return noise


def sort_seeds(seeds, width, height):
    nseeds = [seeds[0]]
    cur_latent = get_latent(seeds[0], width, height)
    seeds.remove(seeds[0])

    while len(seeds)>0:
        min_dist = torch.tensor(99999999.0).to('cuda')
        min_latent = None
        min_seed = None
        for seed in seeds:
            b = get_latent(seed, width, height)
            dist = sum(((cur_latent - b)**2).reshape(16384))
            if dist < min_dist:
                min_dist = dist
                min_latent = b
                min_seed = seed

        print(min_seed)
        nseeds.append(min_seed)
        seeds.remove(min_seed)
        cur_latent = get_latent(min_seed, width, height)
    return nseeds

@torch.no_grad()
def get_text_embedding(prompt):
    clip_tokenizer = get_model().clip_tokenizer
    clip = get_model().clip
    device = get_model().device

    with autocast(device):
        tokens_conditional = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True)
        embedding_conditional = clip(tokens_conditional.input_ids.to(device)).last_hidden_state
    return embedding_conditional

def slerp(t, v0, v1, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def lerp_latent_and_embedding(latent1, latent2, embedding1, embedding2, t):
    latent = slerp(t, latent1, latent2)
    embedding = torch.lerp(embedding1, embedding2, t)
    return latent, embedding

def print_token(prompt):
    clip_tokenizer = get_model().clip_tokenizer
    tokens = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True).input_ids[0]
    index = 1
    while True:
        word = clip_tokenizer.decode(tokens[index:index+1])
        if not word: break
        if word == '<|endoftext|>': break
        print(index, word)
        index += 1
        if index > 500: break

def sep_token(prompt):
    clip_tokenizer = get_model().clip_tokenizer
    tokens = clip_tokenizer(prompt, padding="max_length", max_length=clip_tokenizer.model_max_length, truncation=True, return_tensors="pt", return_overflowing_tokens=True).input_ids[0]
    words = []
    index = 1
    while True:
        word = clip_tokenizer.decode(tokens[index:index+1])
        if not word: break
        if word == '<|endoftext|>': break
        words.append(word)
        index += 1
        if index > 500: break
    return words

def parse_edit_weights(prompt, prompt_edit, edit_weights):
    if prompt_edit:
        tokens = sep_token(prompt_edit)
    else:
        tokens = sep_token(prompt)
    
    prompt_edit_token_weights=[]
    for tl, w in edit_weights:
        if isinstance(tl, list) or isinstance(tl, tuple):
            pass
        else:
            tl = [tl]
        for t in tl:
            try:
                if isinstance(t, str):
                    idx = tokens.index(t) + 1
                elif isinstance(t, int):
                    idx = t
                prompt_edit_token_weights.append((idx, w))
            except ValueError as e:
                print(f'error {e}')
            
    return prompt_edit_token_weights

def sd(prompt="", prompt_edit=None,
       text_embedding=None,
       edit_weights=[],
       prompt_edit_tokens_start=0.0, prompt_edit_tokens_end=1.0,
       prompt_edit_spatial_start=0.0, prompt_edit_spatial_end=1.0,
       negative_prompt="",
       guidance_scale=7.5, steps=35,
       seed=None, latent=None,
       width=512, height=512,
       init_image=None, init_image_strength=0.5,
       disable_tqdm=True):

    prompt_edit_token_weights = parse_edit_weights(prompt, prompt_edit, edit_weights)

    img = stablediffusion(prompt=prompt, 
                          text_embedding=text_embedding,
                          prompt_edit=prompt_edit, 
                          prompt_edit_token_weights=prompt_edit_token_weights,
                          prompt_edit_tokens_start=prompt_edit_tokens_start,
                          prompt_edit_tokens_end=prompt_edit_tokens_end,
                          prompt_edit_spatial_start=prompt_edit_spatial_start,
                          prompt_edit_spatial_end=prompt_edit_spatial_end,
                          negative_prompt=negative_prompt,
                          guidance_scale=guidance_scale,
                          steps=steps,
                          seed=seed,
                          init_latents=latent,
                          width=width,
                          height=height,
                          init_image=init_image,
                          init_image_strength=init_image_strength,
                          disable_tqdm=disable_tqdm)
    return img
