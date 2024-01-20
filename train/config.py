'''
  0.   Dataset paths 
'''
path_ds_train = '/{path}/tokenized_train_dataset/'
path_ds_valid = '/{path}/tokenized_valid_dataset/'
path_tokenizer = '/{path}/{bert_wordpiece_tokenizer}/vocab.txt'
path_target_model_name = '/{path}/output_dir/model_checkpoint'

sample_ratio = 0.000 # (default): no sampling

'''
  1.   Training Arguments
'''
per_device_train_batch_size = 32
per_device_eval_batch_size = 32

MAX_EPOCH = 20
ES_PATIENCE = 3
LEARNING_RATE = 1e-4

gradient_accumulation_steps = 4
eval_accumulation_steps = 10
eval_save_strategy = 'steps' # ['no', 'epoch', 'steps']
eval_steps = 1000
save_steps = 1000


'''
  2.   Model Parameters
'''
N_POSITIONS = 1024
N_LAYERS = 4
N_HEADS = 8
N_EMBDS = 768
N_CTX = 1024

MAX_LENGTH = N_POSITIONS

