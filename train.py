import os
import re
import torch
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup, BertTokenizer

from modeling import CLEAR
from dataset import CLEARDataset, get_collate_function
from utils import generate_rank, eval_results, run_parse_args, save_model, set_seed

logger = logging.getLogger(__name__)
logging.basicConfig(format = '%(asctime)s-%(levelname)s-%(name)s- %(message)s',
                        datefmt = '%d %H:%M:%S',
                        level = logging.INFO)


def train(args, model):
    """ Train the model """
    tb_writer = SummaryWriter(args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    train_dataset = CLEARDataset(mode="train", args=args)
    train_sampler = SequentialSampler(train_dataset) 
    collate_fn = get_collate_function(mode="train")
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, 
        batch_size=args.train_batch_size, num_workers=args.data_num_workers, pin_memory=True, collate_fn=collate_fn)

    steps_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=steps_total)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", steps_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for epoch_idx, _ in enumerate(train_iterator):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            batch = {k:v.to(args.device) for k, v in batch.items()}
            model.train()
            outputs = model(**batch)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) #不确定原文是否用到

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1
                if args.evaluate_during_training and (global_step % args.training_eval_steps == 0):
                    mrr = evaluate(args, model, mode="dev", prefix="step_{}".format(global_step))
                    tb_writer.add_scalar('dev/MRR@10', mrr, global_step)

                if args.logging_steps > 0 and (global_step % args.logging_steps == 0):
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    cur_loss =  (tr_loss - logging_loss)/args.logging_steps
                    tb_writer.add_scalar('train/loss', cur_loss, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and (global_step % args.save_steps == 0):
                    # Save model checkpoint
                    save_model(model, 'ckpt-{}'.format(global_step), args)


def evaluate(args, model, mode, prefix):
    eval_output_dir = args.eval_save_dir
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)
  
    eval_dataset = CLEARDataset(mode=mode, args=args)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    collate_fn = get_collate_function(mode=mode)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size,
        num_workers=args.data_num_workers, pin_memory=True, collate_fn=collate_fn)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    output_file_path = f"{eval_output_dir}/{prefix}.{mode}.score.tsv"
    with open(output_file_path, 'w') as outputfile:
        for batch, qids, pids in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            with torch.no_grad():
                batch = {k:v.to(args.device) for k, v in batch.items()}
                outputs = model(**batch)
                scores = outputs[0]
                assert len(qids) == len(pids) == len(scores)
                for qid, pid, score in zip(qids, pids, scores):
                    outputfile.write(f"{qid}\t{pid}\t{score}\n")
    
    rank_output = f"{eval_output_dir}/{prefix}.{mode}.rank.tsv"
    generate_rank(output_file_path, rank_output)

    if mode == "dev":
        mrr = eval_results(rank_output)
        return mrr


def main():
    args = run_parse_args()

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logger.warning("Device: %s, n_gpu: %s", device, args.n_gpu)

    # Set seed
    set_seed(args)

    if args.mode == "train":
        load_model_path = args.bert_path
    else:
        assert args.eval_ckpt is not None
        load_model_path = os.path.join(args.model_save_dir, f"ckpt-{args.eval_ckpt}")  #从checkpoint中读取模型

    # New special tokens: [QRY] and [DOC]
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    tokenizer.add_tokens(["[QRY]", "[DOC]"])
    args.qry_id = tokenizer.encode('[QRY]', add_special_tokens=False)[0]
    args.doc_id = tokenizer.encode('[DOC]', add_special_tokens=False)[0]
    args.sep_id = tokenizer.sep_token_id

    config = BertConfig.from_pretrained(load_model_path)
    model = CLEAR.from_pretrained(load_model_path, config=config, args=args)
    model.resize_token_embeddings(len(tokenizer))
    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    # Evaluation
    if args.mode == "train":
        train(args, model)
    else:
        result = evaluate(args, model, args.mode, prefix=f"ckpt-{args.eval_ckpt}")
        print(result)


if __name__ == "__main__":
    main()
