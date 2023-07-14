import os 
import torch
import numpy as np
import json
import clip

from options import option
import models.vqvae as vqvae
import utils.utils_model as utils_model
from utils.evaluate import evaluation
from dataloader.eval_loader import DATALoader
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import sys
import time
from pathlib import Path
from typing import Optional

import lightning as L
import torch

from lit_llama import LLaMA, LLaMAConfig
from lit_llama.lora import lora
from lit_llama.utils import EmptyInitOnDevice, lazy_load
from lit_llama.tokenizer import Tokenizer


args = option.get_args_parser()

def main(
    quantize: Optional[str] = None,
    dtype: str = "float32",
    accelerator: str = "auto"
) -> None:
    os.makedirs(args.out_dir, exist_ok = True)

    ##### ---- Logger ---- #####
    logger = utils_model.get_logger(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    from utils.word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_loader = DATALoader(args.dataname, 'test', 32, w_vectorizer, unit_length=2**args.down_t)

    if args.dataname == 'kit' : 
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'  
        args.nb_joints = 21
    else :
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 22

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Network ---- #####

    ## load clip model and datasets
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    print('Loading VAE')
    vae = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        512,
                        args.code_dim,
                        args.output_emb_width,
                        2,
                        args.stride_t,
                        args.width,
                        3,
                        args.dilation_growth_rate)
    resume_pth = f"./checkpoints/pretrained_vqvae/{args.dataname}.pth"
    ckpt = torch.load(resume_pth, map_location='cpu')
    vae.load_state_dict(ckpt['net'], strict=True)
    vae = vae.cuda().eval()
    print('Loading VAE Done')

    lora_path = Path(args.lora_path)
    print('Load finetuned model from:', lora_path)
    pretrained_path = Path(f"./checkpoints/lit-llama/{args.pretrained_llama}/lit-llama.pth")
    tokenizer_path = Path("./checkpoints/lit-llama/tokenizer.model")
    
    assert lora_path.is_file()
    assert pretrained_path.is_file()
    assert tokenizer_path.is_file()

    if quantize is not None:
        raise NotImplementedError("Quantization in LoRA is not supported yet")

    fabric = L.Fabric(accelerator=accelerator, devices=1)

    dt = getattr(torch, dtype, None)
    if not isinstance(dt, torch.dtype):
        raise ValueError(f"{dtype} is not a valid dtype.")
    dtype = dt

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with EmptyInitOnDevice(
        device=fabric.device, dtype=dtype, quantization_mode=quantize
    ), lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, enabled=True):
        # model = LLaMA(LLaMAConfig())  # TODO: Support different model sizes
        config = LLaMAConfig.from_name(args.pretrained_llama)
        model = LLaMA(config)

    # 1. Load the pretrained weights
    pretrained_checkpoint = lazy_load(pretrained_path)
    model.load_state_dict(pretrained_checkpoint, strict=False)

    # 2. Load the fine-tuned LoRA weights
    lora_checkpoint = lazy_load(lora_path)
    model.load_state_dict(lora_checkpoint, strict=False)

    print(f"Time to load model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

    model.eval()
    model = fabric.setup_module(model)
    tokenizer = Tokenizer(tokenizer_path)

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    repeat_time = 3
            
    for _ in range(repeat_time):
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger = evaluation(val_loader, vae, model, logger, tokenizer, eval_wrapper=eval_wrapper, instruction=args.prompt)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)

    print('final result:')
    print('fid: ', sum(fid)/repeat_time)
    print('div: ', sum(div)/repeat_time)
    print('top1: ', sum(top1)/repeat_time)
    print('top2: ', sum(top2)/repeat_time)
    print('top3: ', sum(top3)/repeat_time)
    print('matching: ', sum(matching)/repeat_time)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
    logger.info(msg_final)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore", 
        message="ComplexHalf support is experimental and many operators don't support it yet"
    )
    main()