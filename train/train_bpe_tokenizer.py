import sys
import os

from datetime import datetime
from hyspark import Hyspark
from spark_hive_utils import *

from tokenizers import ByteLevelBPETokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tokenizers.processors import ByteLevel as ByteLevelProcessor

sys.path.append("../utils")

def get_employee_id():
    from os import getcwd
    from re import search
    return str(search(r'.+(\d{6}).?', getcwd()).group(1))


now= datetime.now()
curr_time = now.strftime('%H:%M:%S')
employee_id = get_employee_id()
hs = HySpark(f'{employee_id}_seqdata_tokenizer_LLM_{curr_time}',
             mem_per_core=8, instance='general')

hc, sc, ss = hs.hive_context, hs.spark_context, hs.spark_session
check_hive_available(hc)


# load data
ps_df = hc.sql('SELECT * FROM db_name.llm_uniq_evnt_parsed_v1')
df = df_as_pandas_with_pyspark(ps_df)


tokenizer = ByteLevelBPETokenizer(add_prefix_space=False, lowercase=False)
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()
tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)
tokenizer.normalizer = NFKC()

tokenizer.train_from_iterator(iter(df), vocab_size=int(1.1e5),
                              show_progress=False, special_tokens=["<|endoftext|>", "<|padding|>"])

os.makedirs('/myhome/mydir/mytokenizer/bpe_tokenizer/', exist_ok=True)
tokenizer.save('/myhome/mydir/mytokenizer/bpe_tokenizer/tokenizer.json', pretty=True)
tokenizer.save_model('/myhome/mydir/mytokenizer/bpe_tokenizer/')
