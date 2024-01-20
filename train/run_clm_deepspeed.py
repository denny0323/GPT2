# deepspeed --include localhost: 2, 3, 4 -- master_poart 4551 run_clm_deepspeed.py --deepspeed_config ds_config_zeros.json

import os
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from datasets import load_from_disk

import sys
sys.path.append('../utils')

import config
from run_utils import *

from transformers import BertTokenizer # WordPiece tokenizer사용
from transformers import TrainingArguments
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from transformers import Trainer, EarlyStoppingCallback

n_scores = 20

os.environ["NUMEXPR_NUM_THREADS"] = str(n_cores)
os.environ["OPENBLAS_NUM_THREADS"] = str(n_cores)
os.environ["OMP_NUM_THREADS"] = str(n_cores)
os.environ["MKL_NUM_THREADS"] = str(n_cores)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_cores)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
os.environ["NCCL_DEBUG"] = "INFO"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------* main *--------------- #
lm_datasets_train = load_from_disk(config.path_ds_train)

if config.sample_ratio:
    row_nums = lm_datasets_train.shape[0]
    lm_datasets_train = lm_datasets_train.shuffle()
    lm_datasets_train = lm_datasets_train.select(list(range(0, int(row_nums * config.sample_ratio))))
lm_datasets_valid = load_from_disk(config.path_ds_valid)


tokenizer = BertTokenizer(config.path_tokenizer,
                          padding='max_length', truncation=True,
                          max_len=config.MAX_LENGTH, add_special_tokens=False,
                          do_lower_case=False, strip_accents=False)

# tokenizer = GPT2Tokenizer(vocab_file=config.gpt2_tokenizer_vocab_file_path,
#                           merges_file=config.gpt2_tokenizer_merges_file_path,
#                           padding='max_length', truncation=True,
#                           max_len=config.MAX_LENGTH)
# tokenizer.pad_token = tokenizer.eos_token
#
# special_token_dict = {
#     'eos_token': '<|eos|>',
#     'bos_token': '<|bos|>',
#     'pad_token': '<|eos|>',
#     'unk_token': '<|unk|>'
# }
# tokenizer.add_special_tokens(special_token_dict)

def group_texts(examples):
    context_length = config.MAX_LENGTH

    from collections import defaultdict
    result = defaultdict(list)
    for input_ids, attention_mask in zip(examples['input_ids'], examples['attention_mask']):
        total_length = len(input_ids)

        for i in range(0, total_length, context_length):
            input_ids_chunk = input_ids[i:i+context_length]
            attention_mask_chunk = attention_mask[i:i+context_length]
            token_type_ids = tokenizer.create_token_type_ids_from_sequences(input_ids_chunk)

            result['input_ids'].append(input_ids_chunk)
            result['attention_mask'].append(attention_mask_chunk)
            result['token_type_ids'].append(token_type_ids)
            del input_ids_chunk, attention_mask_chunk, token_type_ids

    return result

def decode_retokenize(examples):
    decoded = {'evnt': [tokenizer.decode(evnt) for evnt in examples['evnt']]}
    return tokenizer(decoded['evnt'], truncation=True, add_special_tokens=False, max_length=config.MAX_LENGTH)


lm_datasets_train = lm_datasets_train.rename_column('input_ids', 'evnt')
lm_datasets_valid = lm_datasets_valid.rename_column('input_ids', 'evnt')

lm_datasets_train.set_format("torch")
lm_datasets_valid.set_format("torch")


config_gpt = GPT2Config(
    architectures='GPT2LMHeadModel',
    vocab_size=tokenizer.vocab_size,
    n_positions=config.N_POSITIONS,
    n_embd=config.N_EMBDS,
    n_layer=config.N_LAYERS,
    n_head=config.N_HEADS,
    n_inner=None,
    activation_function='gelu_new',
    n_ctx=config.N_CTX,
    max_length=config.N_POSITIONS,
    bos_token_id=tokenizer.cls_token_id,
    eos_token_id=tokenizer.sep_token_id,
    attn_pdrop=0.1,
    embd_pdrop=0.1,
    initializer_range=0.02,
    layer_norm_epsilon=1e-5,
    resid_pdrop=0.1
)


model_train = GPT2LMHeadModel(config_gpt)


training_args = TrainingArguments(
    output_dir=config.path_target_model_name,
    do_train=True,
    do_eval=True,
    fp16=True,
    fp16_opt_level="02",  # mixed precision mode
    fp16_full_eval=True,
    num_train_epochs=config.MAX_EPOCH,
    learning_rate=config.LEARNING_RATE,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    eval_accumulation_steps=config.eval_accumulation_steps,
    per_device_train_batch_size=config.per_device_train_batch_size,
    per_device_eval_batch_size=config.per_device_eval_batch_size,
    evaluation_strategy=config.eval_save_strategy,
    eval_steps=config.eval_steps,
    save_strategy=config.eval_save_strategy,
    save_steps=config.save_steps,
    save_total_limit=5,
    load_best_model_at_end=True,
    weight_decay=0.01,
    seed=42
)


from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)


trainer = Trainer(
    model=model_train,
    args=training_args,
    train_dataset=lm_datasets_train,
    eval_dataset=lm_datasets_valid,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_gpt2_hf,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=config.ES_PATIENCE)]
)

trainer.train()
# trainer.train(resume_from_checkpoint=config.path_target_model_name+'/checkpoint-00000')




