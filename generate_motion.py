import os
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import lightning as L
import torch
import numpy as np

import models.vqvae as vqvae
from generate import generate
from lit_llama import Tokenizer, LLaMA, LLaMAConfig
from lit_llama.lora import lora
from lit_llama.utils import EmptyInitOnDevice, lazy_load
from scripts.prepare_motion import generate_prompt
from options import option
import imageio
from utils.evaluate import plot
from visualization.render import render
warnings.filterwarnings('ignore')

args = option.get_args_parser()


def main(
    quantize: Optional[str] = None,
    dtype: str = "float32",
    max_new_tokens: int = 200,
    top_k: int = 200,
    temperature: float = 0.8,
    accelerator: str = "auto",
) -> None:
    lora_path = Path(args.lora_path)
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

    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)
    print ('loading checkpoint from {}'.format(args.vqvae_pth))
    ckpt = torch.load(args.vqvae_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.cuda()

    print("Loading model ...", file=sys.stderr)
    t0 = time.time()
    with EmptyInitOnDevice(
        device=fabric.device, dtype=dtype, quantization_mode=quantize
    ), lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, enabled=True):
        config = LLaMAConfig.from_name(args.pretrained_llama)
        model = LLaMA(config)
        # model = LLaMA(LLaMAConfig())  # TODO: Support different model sizes

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
    sample = {"instruction": args.prompt, "input": args.input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, bos=True, eos=False, device=model.device)

    t0 = time.perf_counter()
    output = generate(
        model,
        idx=encoded,
        max_seq_length=max_new_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id
    )

    output = tokenizer.decode(output)
    output = output.split("### Response:")[1].strip()

    t = time.perf_counter() - t0

    print(f"\n\nTime for inference: {t:.02f} sec total, {max_new_tokens / t:.02f} tokens/sec", file=sys.stderr)
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB", file=sys.stderr)

    tokens = torch.tensor([int(token) for token in output.split(',')]).cuda()
    generated_pose, img = plot(tokens, net, args.dataname)
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'demo.npy'), generated_pose)
    imageio.mimsave(os.path.join(args.out_dir, 'demo.gif'), np.array(img), fps=20)
    if args.render:
        print("Rendering...")
        render(generated_pose, 'demo', outdir=args.out_dir)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
