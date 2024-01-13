# Usage : torchrun inference_gpt2_ddp.py --nproc_per_node={gpu 개수}
# coding: utf-8

import os, gc
import torch
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR) # evaluation시 error meassage만 출력
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)


import sys
sys.path.append('../{file_script_path1}') # 기타 필요 script load경로
sys.path.append('../{file_script_path2}')
sys.path.append('../{file_script_path3}')
sys.path.append('../{file_script_path4}')
sys.path.append('../{file_script_path5}')

from config import *

## CPU
n_cores = 40
os.environ['NUMEXPR_NUM_THREADS'] = str(n_cores)
os.environ['OPENBLAS_NUM_THREADS'] = str(n_cores)
os.environ['OMP_NUM_THREADS'] = str(n_cores)
os.environ['MKL_NUM_THREASD'] = str(n_cores)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(n_cores)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
os.environ['NCCL_DEBUG'] = 'INFO'

## GPU
use_gpu_num = [0, 1, 2. 3]

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(num) for num in use_gpu_num])
os.environ["CUDA_LAUNCH_BLOCKING"] = '1'

def set_tempdir():
    return 'my/local/temp/dir'

from datasets import table
table.tempfile._get_default_tempdir = set_tempdir
table.tempfile.tempdir = 'my/local/temp/dir'

import datasets
datasets.config.MAX_TABLE_NBYTES_FOR_PICKLING /= 10

from src.eval_tool import Evaluator

import datetime
import torch.multiprocessing as mp

from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer(
    "tokenizer_path",
    add_special_tokens=False,
    do_lower_cas=False,
    strip_accents=False
)

seperator = ['[EOHW]', '[EOW]']
tokenizer.add_tokens(seperator)

week_tokens = [f'[{i}' for i in range(1, 53)]
tokenizer.add_tokens(week_tokens)

################### load inputs ##################
interm_result = pd.read_csv('output.csv', sep='\t')
to_test_df = pd.read_picke('test_df.pkl')

# csno_filtering : 중간 결과에서 다룬 데이터 제외
# to solve csno type mismatch (interm_result.csno:int64 != to_test_df.csno:object)
to_test_df.csno = to_test_df.csno.astype(int)
to_test_df = to_test_df[~to_test_df.csno.isin(interm_result.csno.drop_duplicates())]


#################### load Model ###################
from transformers import GPT2Config, GPT2LMHeadModel
pt_gpt_ckpt = 'trained_gpt_ckpt/checkpoint-000000'
config_gpt = GPT2Config.from_json_file(pt_gpt_ckpt+'/config.json')

model = GPT2LMHeadModel(config_gpt)
model.load_state_dict(torch.load(pt_gpt_ckpt+'/pytorch_model.bin'))




def run_inference(rank, world_size, model):

    batch_size = 2
    top_k = 10
    log_interval = 500

    days = 30
    torch.backends.cudnn.benchmark = True

    ###################### DDP settings #######################
    torch.distributed.init_process_group(backend="nccl",
                                         init_method='env://',
                                         rank=rank,
                                         wordl_size=world_size,
                                         timeout=datetime.timedelta(0, days*24*3600))

    print('\nrank', rank)
    print('word_size', world_size)

    model = model.to(rank)

    resume_split = int(len(to_test_df)//world_size)

    evaluator = Evaluator(model, tokenizer, save_type='csv')

    with torch.no_grad():
        if torch.distributed.get_rank() == 0:
            print(f'torch.distributed.get_rank() : {torch.distributed.get_rank()}')
            evaluator.do_eval(batch_size=batch_size,
                              to_test_df=to_test_df[:resume_split],
                              top_k=top_k,
                              rank=torch.distributed.get_rank(),
                              log_interval=log_interval)

        elif torch.distributed.get_rank() == 1:
            print(f'torch.distributed.get_rank() : {torch.distributed.get_rank()}')
            evaluator.do_eval(batch_size=batch_size,
                              to_test_df=to_test_df[resume_split:2*resume_split],
                              top_k=top_k,
                              rank=torch.distributed.get_rank(),
                              log_interval=log_interval)

        elif torch.distributed.get_rank() == 2:
            print(f'torch.distributed.get_rank() : {torch.distributed.get_rank()}')
            evaluator.do_eval(batch_size=batch_size,
                              to_test_df=to_test_df[2*resume_split:3*resume_split],
                              top_k=top_k,
                              rank=torch.distributed.get_rank(),
                              log_interval=log_interval)

        else:
            print(f'torch.distributed.get_rank() : {torch.distributed.get_rank()}')
            evaluator.do_eval(batch_size=batch_size,
                              to_test_df=to_test_df[3*resume_split:],
                              top_k=top_k,
                              rank=torch.distributed.get_rank(),
                              log_interval=log_interval)

        torch.cuda.empty_cache()
        gc.collect()


def main():
    world_size = len(use_gpu_num)

    ### main process
    mp.spawn(run_inference,
             args=(world_size, model),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    main()