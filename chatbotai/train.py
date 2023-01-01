from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import glob, logging, os, pickle, random, re, torch, pandas as pd, numpy as np
from typing import Dict, List, Tuple
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, DistributedSampler
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm, trange

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)

def construct_conv(row, tokenizer, eos = True):
    flatten = lambda l: [item for sublist in l for item in sublist]
    conv = list(reversed([tokenizer.encode(x) + [tokenizer.eos_token_id] for x in row]))
    conv = flatten(conv)
    return conv

def load_and_cache_examples(args, tokenizer, df_trn):
    return ConversationDataset(tokenizer, args, df_trn)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

class ConversationDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, df, block_size=512):

        block_size = block_size - (tokenizer.model_max_length - tokenizer.max_len_single_sentence)
        directory = args.cache_dir
        cached_features_file = os.path.join(directory, args.model_type + "_cached_lm_" + str(block_size))

        logger.info("Creating features from dataset file at %s", directory)
        self.examples = []
        for _, row in df.iterrows():
            conv = construct_conv(row, tokenizer)
            self.examples.append(conv)

        logger.info("Saving features into cached file %s", cached_features_file)
        with open(cached_features_file, "wb") as handle:
            pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)

def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate, drop_last = True
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    model = model.module if hasattr(model, "module") else model  
    model.resize_token_embeddings(len(tokenizer))

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    logger.info("*** Running trainng, Num examples = %d, Num Epochs = %d ***", len(train_dataset), args.num_train_epochs)

    global_step, epochs_trained = 0, 0
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility
    alltimelowestloss = 99999999
    count = 0
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            inputs, labels = (batch, batch)
            if inputs.shape[1] > 1024: continue
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            print(loss.item())

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step() 
                model.zero_grad()
                global_step += 1

                if (loss.item() < alltimelowestloss):
                    model_to_save = (model.module if hasattr(model, "module") else model)  
                    model_to_save.save_pretrained(args.output_dir)


                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss
        count += 1
        

    tb_writer.close()

    return global_step, tr_loss / global_step

logger = logging.getLogger(__name__)

class Args():
    def __init__(self):
        self.output_dir = f'output-medium1'
        self.model_type = 'gpt2'
        self.model_name_or_path = f'microsoft/DialoGPT-medium'
        self.config_name = f'microsoft/DialoGPT-medium'
        self.tokenizer_name = f'microsoft/DialoGPT-medium'
        self.cache_dir = 'cached'
        self.block_size = 512
        self.per_gpu_train_batch_size = 4
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.num_train_epochs = 1
        self.max_steps = -1
        self.warmup_steps = 0
        self.logging_steps = 1000
        self.save_total_limit = None
        self.seed = 42
        self.local_rank = -1





df = pd.read_csv("userdataset.csv")

def main_train(df_trn):
    args = Args()
    
    device = torch.device("cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    set_seed(args) 

    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium", cache_dir=args.cache_dir)
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium", from_tf=False, config=config, cache_dir=args.cache_dir)
    model.to(args.device)
    train_dataset = load_and_cache_examples(args, tokenizer, df_trn)
    global_step, tr_loss = train(args, train_dataset, model, tokenizer)

    os.makedirs(args.output_dir, exist_ok=True)

def real_time(tokenizer, model, args, train_dataset):
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()
    
    args.train_batch_size = args.per_gpu_train_batch_size 


    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate, drop_last = True
    )


    model.resize_token_embeddings(len(tokenizer))


    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    epochs_trained = 0

    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )

    set_seed(args)  
    alltimelowestloss = 99999999
    epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    for step, batch in enumerate(epoch_iterator):
            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                model.zero_grad()

                if (loss.item() < alltimelowestloss):
                    model.save_pretrained(args.output_dir)
    print("realtimedone")
        

    tb_writer.close()


def beginrealtime(tokenizer, model):
    args = Args()
    device = torch.device("cpu")
    args.device = device
    model.to(args.device)

    train_dataset = load_and_cache_examples(args, tokenizer, df)
    os.makedirs(args.output_dir, exist_ok=True)
    real_time(tokenizer, model, args, train_dataset)

