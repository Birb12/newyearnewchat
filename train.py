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
