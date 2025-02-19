from diffusers import StableDiffusionXLPipeline,AutoencoderKL
import torch
from datetime import datetime


BLOCKS = {
    'style': ['up_blocks.0.attentions.1'],
}


def is_belong_to_blocks(key, blocks):
    try:
        for g in blocks:
            if g in key:
                return True
        return False
    except Exception as e:
        raise type(e)(f'failed to is_belong_to_block, due to: {e}')


def filter_lora(state_dict, blocks_):
    try:
        return {k: v for k, v in state_dict.items() if is_belong_to_blocks(k, blocks_)}
    except Exception as e:
        raise type(e)(f'failed to filter_lora, due to: {e}')


def scale_lora(state_dict, alpha):
    try:
        return {k: v * alpha for k, v in state_dict.items()}
    except Exception as e:
        raise type(e)(f'failed to scale_lora, due to: {e}')


def get_target_modules(unet, blocks=None):
    try:
        if not blocks:
            blocks = [('.').join(blk.split('.')[1:]) for blk in BLOCKS['content'] + BLOCKS['style']]

        attns = [attn_processor_name.rsplit('.', 1)[0] for attn_processor_name, _ in unet.attn_processors.items() if
                 is_belong_to_blocks(attn_processor_name, blocks)]

        target_modules = [f'{attn}.{mat}' for mat in ["to_k", "to_q", "to_v", "to_out.0"] for attn in attns]
        return target_modules
    except Exception as e:
        raise type(e)(f'failed to get_target_modules, due to: {e}')




pipeline = StableDiffusionXLPipeline.from_pretrained(
    r"C:/Users/Admin/.cache/modelscope/hub/AI-ModelScope/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16,
    variant="fp16", use_safetensors=True,
).to("cuda")


state_dict1,_ = pipeline.lora_state_dict(r"D:\0617\MagicMix-main\unzip\3D2_400\unziplora.pt")


# style_B_LoRA = state_dict1
style_B_LoRA = filter_lora(state_dict1, BLOCKS['style'])

# 给所有key加上前缀 'unet_'
modified_dict = {f'unet.{key}': value for key, value in style_B_LoRA.items()}
#
# # 删除所有key包含'alora'或'clora'的记录
# modified_dict = {key: value for key, value in modified_dict.items() if 'c_lora' not in key and 'column' not in key and 'linear' not in key}

modified_dict = {key: value for key, value in modified_dict.items() if 'c_lora' not in key and 'column_mask' not in key and 'linear' not in key}


# 使用字典推导式替换包含's_lora_A'的部分
modified_dict = {key.replace('s_lora_A', 'lora.down'): value for key, value in modified_dict.items()}
# 使用字典推导式替换包含's_lora_A'的部分
modified_dict = {key.replace('s_lora_B', 'lora.up'): value for key, value in modified_dict.items()}
# # Load
pipeline.load_lora_into_unet(modified_dict, None, pipeline.unet)


import torch

dict1 = torch.load(r'D:\0617\MagicMix-main\unzip\3D2_400\token_embeddings_1.pt')

dict2 = torch.load(r'D:\0617\MagicMix-main\unzip\3D2_400\token_embeddings_2.pt')


pipeline.load_textual_inversion(dict1["embeddings"], token=['[C]', '[S]'], text_encoder=pipeline.text_encoder, tokenizer=pipeline.tokenizer)
pipeline.load_textual_inversion(dict2["embeddings"], token=['[C]', '[S]'], text_encoder=pipeline.text_encoder_2, tokenizer=pipeline.tokenizer_2)


# # prompt is passed to OAI CLIP-ViT/L-14
# prompt = "A dog in [S] style."
# prompt_list=["A Tooth icon","A Database icon","A Book icon","A monitor icon","A dog icon","A cow icon","An elephant icon"]

# prompt_list=["A pig icon","A bird icon","A cat icon","A tiger icon","A Cafe icon","A Ruler icon","A City Hall icon"]

# prompt_list=["An Apple icon","A Banana icon","An Orange fruit icon","A Grape icon","A Pencil icon","A Notebook icon","A Eraser icon"]

# prompt_list=["An Apple","A Banana","An Orange fruit","A Grape","A Pencil","A Notebook","A Eraser"]

prompt_list=["A Tooth","A Database","A Book","A monitor","A dog","A cow","An elephant"]

prompt = [i + " in the style of [S]" for i in prompt_list if i]

seed = 42
#
# # Generate
images = pipeline(prompt,  genetator=torch.Generator("cpu").manual_seed(seed)).images

for i, img in enumerate(images):
    print(i)
    # 获取当前时间并格式化为字符串
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    img.save(fr'D:\0709\B-LoRA-main\img_output\{prompt[i]}_{seed}_{timestamp}.png')
