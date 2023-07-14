import os
import time

import lightning as L
import numpy as np
import torch
import clip
from lit_llama.lora import mark_only_lora_as_trainable, lora, lora_state_dict
from lit_llama.model import LLaMA, LLaMAConfig
from lit_llama.tokenizer import Tokenizer
from dataloader.eval_loader import DATALoader
from utils.evaluate import evaluation
from utils.word_vectorizer import WordVectorizer
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.vqvae as vqvae
from options import option
import utils.utils_model as utils_model
from torch.utils.tensorboard import SummaryWriter
import json

args = option.get_args_parser()
gradient_accumulation_steps = args.batch_size // args.micro_batch_size
max_iters = 50000 * 3 // args.micro_batch_size


def main():
    fabric = L.Fabric(accelerator="cuda", devices=1, precision="bf16-mixed")
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(args.out_dir, exist_ok=True)

    train_data, val_data = load_datasets()

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_loader = DATALoader(args.dataname, 'val', 32, w_vectorizer, unit_length=2**args.down_t)

    if args.dataname == 'kit' : 
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'  
        args.nb_joints = 21
    else :
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 22
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

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

    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    config = LLaMAConfig.from_name(args.pretrained_llama)
    config.block_size = args.block_size

    checkpoint = torch.load(f"./checkpoints/lit-llama/{args.pretrained_llama}/lit-llama.pth")
    tokenizer = Tokenizer("./checkpoints/lit-llama/tokenizer.model")

    with fabric.device, lora(r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout, enabled=True):
        torch.set_default_tensor_type(torch.HalfTensor)
        model = LLaMA(config).bfloat16()
        torch.set_default_tensor_type(torch.FloatTensor)
        # strict=False because missing keys due to LoRA weights not contained in checkpoint state
        model.load_state_dict(checkpoint, strict=False)
        if args.resume_pth:
            checkpoint = torch.load(args.resume_pth)
            model.load_state_dict(checkpoint, strict=False)
    
    mark_only_lora_as_trainable(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model, optimizer = fabric.setup(model, optimizer)
    train(fabric, model, optimizer, train_data, val_data, args.out_dir, logger, writer)

    # Save the final LoRA checkpoint at the end of training
    checkpoint = lora_state_dict(model)
    fabric.save(os.path.join(args.out_dir, "lit-llama-lora-finetuned.pth"), checkpoint)

    # Evaluation on validation set
    evaluation(val_loader, net, model, logger, tokenizer, eval_wrapper=eval_wrapper, instruction=args.prompt)


def train(
    fabric: L.Fabric,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: np.ndarray,
    val_data: np.ndarray,
    out_dir: str,
    logger,
    writer
) -> None:
    """The training loop.

    Loosely based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
    """
    step_count = 0

    for iter_num in range(max_iters):

        if step_count <= args.warmup_steps:
            # linear warmup
            lr = args.learning_rate * step_count / args.warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        t0 = time.time()

        input_ids, targets = get_batch(fabric, train_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        fabric.backward(loss)

        if (iter_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            if step_count % args.eval_interval == 0:
                val_loss = validate(fabric, model, val_data)
                writer.add_scalar('./Val', val_loss, step_count)
                logger.info(f"step {iter_num}: val loss {val_loss:.4f}")
                fabric.barrier()

            if step_count % args.save_interval == 0:
                print(f"Saving LoRA weights to {out_dir}")
                # We are only saving the LoRA weights
                # TODO: Provide a function/script to merge the LoRA weights with pretrained weights
                checkpoint = lora_state_dict(model)
                fabric.save(os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth"), checkpoint)

        dt = time.time() - t0
        if iter_num % args.log_interval == 0:
            writer.add_scalar('./Train', loss, iter_num)
            logger.info(f"iter {iter_num}: loss {loss.item():.4f}, time: {dt*1000:.2f}ms")


@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_data: np.ndarray) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    losses = torch.zeros(args.eval_iters)
    for k in range(args.eval_iters):
        input_ids, targets = get_batch(fabric, val_data)
        logits = model(input_ids)
        loss = loss_fn(logits, targets)
        losses[k] = loss.item()
    out = losses.mean()

    model.train()
    return out.item()

def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss
    

def get_batch(fabric: L.Fabric, data: list):
    ix = torch.randint(len(data), (args.micro_batch_size,))

    input_ids = [data[i]["input_ids"].type(torch.int64) for i in ix]
    labels = [data[i]["labels"].type(torch.int64) for i in ix]

    max_len = max(len(s) for s in input_ids)

    def pad_left(x, pad_id):
        # pad right based on the longest sequence
        n = max_len - len(x)
        return torch.cat((torch.full((n,), pad_id, dtype=x.dtype), x))
    
    # def pad_right(x, pad_id):
    #     # pad right based on the longest sequence
    #     n = max_len - len(x)
    #     return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))

    x = torch.stack([pad_left(x, pad_id=0) for x in input_ids])
    y = torch.stack([pad_left(x, pad_id=-1) for x in labels])
    x, y = fabric.to_device((x.pin_memory(), y.pin_memory()))
    return x, y


def load_datasets():
    print('Load data from:', args.data_dir)
    train_data = torch.load(os.path.join(args.data_dir, "train.pt"))
    val_data = torch.load(os.path.join(args.data_dir, "val.pt"))
    return train_data, val_data


if __name__ == "__main__":
    # Uncomment this line if you see an error: "Expected is_sm80 to be true, but got false"
    # torch.backends.cuda.enable_flash_sdp(False)
    torch.set_float32_matmul_precision("high")
    
    # from jsonargparse.cli import CLI

    # args = option_trans.get_args_parser()
    # args.dataname = 't2m'
    # args.out_dir = 'out/lora/mydataset_v3'
    # logger = utils_model.get_logger(args.out_dir)
    # writer = SummaryWriter(args.out_dir)
    # logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    # CLI(main)
    main()