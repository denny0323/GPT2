import gc
import torch
import numpy as np
import pandas as pd
import pickle as pkl

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR) # evaluation시 error meassage만 출력
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)

import sys
sys.path.append('../{file_script_path1}') # 기타 필요 script load경로
sys.path.append('../{file_script_path2}')

from datetime import datetime
from hyspark import Hyspark # spark custom ver. file
from spark_hive_table_utils import check_hive_available, df_as_pandas_with_pyspark # custom spark utils file
#from IPython.core.display import display

from test_utils import modified_precision # my utils
from typing import List, Tuple, Union, Optional
from transformers import BatchEncoding

from sklearn.metrics import classification_report, accuracy_score

from metric import Metric # my tools
from collections import defaultdict

gc_threshold = tuple(np.asarray(gc.get_threshold())*0.8)
gc.set_threshold(*tuple(map(int, gc_threshold)))

from packer import sai_packer
from pyspark.sql.types import *
from tqdm import tqdm

# for AUC metric in keras
import tensorflow as tf
from tensorflow.keras.metrics import AUC
gpus = tf.config.experimental.list_physical_device('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)



##############################################
########     0.  basic functions      ########
##############################################


# for spark session (hyspark)
def get_employee_ids():
    from os import getcwd
    from re import search
    return str(search(r'.+(\d{6}).?', getcwd()).group(1))


##############################################
########        1.  main class        ########
##############################################
class Evaluator:
    def __init__(self, model, tokenizer, save_type='hive'):
        self.model = model
        self.model.config.torch_dtype = self.model.dtype
        self.model.eval()

        self.tokenizer = tokenizer
        self.tokenizer.padding_side = 'left'

        self.save_type = save_type

        # for target class
        self.target_mccb = {'mccb1': 1,
                            'mccb2': 2,
                            'mccb3': 3,
                            ...
                            'mccb10': 10}

        # for mapping dictionary target mrch:mccb
        self.dictionary_path = ''
        self.dictionary_file = '.pkl'

        self.bad_word_ids = [[1], [3]]
        self.max_new_tokens = 9
        self.min_new_tokens = 3
        self.num_beams = 3

        self.seperator = ['[EOHW]', '[EOW]'] # my custom [SEP] token (=|endoftext|)
        self.week_tokens = [f'[{i}]' for i in range(1, 53)]

        self.now = datetime.now()
        self.today = self.now.strftime("%Y%m%d")

        self.employee_id = get_employee_id()
        self.hyspark_param = {
            'app_name': f'GPT pred session by ({self.employee_id} >>> {self.today}',
            'instance': 'general',
            'mem_per_core': 11,
            'verbose': 0,
        }



    def load_mapping_dictionary(
            self,
            dictionary_path=None,
            dictionary_file=None
    ):
        '''mapping dictionary load하는 함수'''

        if dictionary_path is None:
            dictionary_path = self.dictionary_path

        if dictionary_file is None:
            dictionary_file = self.dictionary_file

        with open(dictionary_path+dictionary_file, 'rb') as f:
            self.mrch_to_mccb = pkl.load(f)


    def preprocess_dictionary(self):
        '''
        mapping dictionary를 pre-processing하는 함수
        multi mccb인 mrch에 대하여 target class만 남겨두는 함수
        '''
        for mrch, mccb in self.mrch_to_mccb.items():
            if len(mccb) == 1:
                self.mrch_to_mccb[mrch] = mccb[-1]
                continue

            elif len(mccb) > 1:
                inter_w_target_mccb = set(self.target_mccb.keys()).intersection(set(mccb))
                if len(inter_w_target_mccb):
                    self.mrch_to_mccb[mrch] = list(inter_w_target_mccb)[0]
                else:
                    self.mrch_to_mccb[mrch] = mccb[0]


    @torch.autocast(device_type='cuda')
    def compute_transition_scores(
            self,
            sequences: torch.Tensor,
            scores: Tuple[torch.Tensor],
            beam_indices: Optional[torch.Tensor] = None,
            normalize_logits: bool = False,
    ) -> torch.Tensor:
        '''
        @source : <huggingface>
        @url : https://github.com/huggingface/transformers/blob/v4.36.1/src/transformers/generation/utils.py#L1141
        transformers의 버전이 낮아 해당 함수를 불러서 post-processing함
        '''

        # 1. In absence of `beam_indices`, we can assume that we come from e.g. greedy search, which is equivalent
        # to a beam search approach were the first (and only) beam is always selected
        if beam_indices is None:
            beam_indices = torch.arange(scores[0].shape[0]).view(-1, 1)
            beam_indices = beam_indices.expand(-1, len(scores))

        # 2. reshape scores as [batch_size*vocab_size, # generation steps] with # generation steps being
        # seq_len - input_length
        scores = torch.stack(scores).reshape(len(scores), -1).transpose(0, 1)

        # 3. Optionally normalize the logits (across the vocab dimension)
        if normalize_logits:
            scores = scores.reshape(-1, self.config.vocab_size, scores.shape[-1])
            scores = torch.nn.functional.log_softmax(scores, dim=1)
            scores = scores.reshape(-1, scores.shape[-1])

        # 4. cut beam_indices to longest beam length
        beam_indices_mask = beam_indices < 0
        max_beam_length = (1 - beam_indices_mask.long()).sum(-1).max()
        beam_indices = beam_indices.clone()[:, :max_beam_length]
        beam_indices_mask = beam_indices_mask[:, :max_beam_length]

        # 5. Set indices of beams that finished early to 0; such indices will be masked correctly afterwards
        beam_indices[beam_indices_mask] = 0

        # 6. multiply beam_indices with vocab size to gather correctly from scores
        beam_sequence_indices = beam_indices * len(self.tokenizer) # 본래 코드는 self.config.vocab_size이나,
                                                                   # 새로운 seperator를 정의하여 add token하여 vocab_size가 커짐
                                                                   # -> config.vocab_size에서는 added token을 count하지 못함

        # 7. Define which indices contributed to scores
        cut_idx = sequences.shape[-1] - max_beam_length
        indices = sequences[:, cut_idx:] + beam_sequence_indices

        # 8. Compute scores
        transition_scores = scores.gather(0, indices)

        # 9. Mask out transition_scores of beams that stopped early
        transition_scores[beam_indices_mask] = 0

        return transition_scores


    def token_to_word_with_transition_prob(
            self,
            generated_id: torch.Tensor,
            transition_probs: torch.Tensor,
    ) -> list:
        '''
        token을 word로 치환하는 함수

        <부가 기능>
        1. word별 beam probability도출
        2. seperator까지 끊기

        Parameters:
            generated_id : dim = (batch_size, top_k, generation_step(=num_tokens_to_generate)
            transition_probs : dim = batch_size, top_k, generation_step(=num_tokens_to_generate)
        '''

        generated_word = []
        generated_word_prob = []
        generated_id_to_token = self.tokenizer.convert_ids_to_tokens(generated_id)

        # sep token이 있는 경우, 이 token 뒤의 내용은 절삭함
        mustbeprocessed = True if len(set(generated_id_to_token).intersection(set(self.week_tokens))) else False

        word = ''
        word_p = 0.0
        for token, p in zip(generated_id_to_token, transition_probs):
            if '##' in token:
                word += token
                word_p *= p

            else:
                if word != '':
                    generated_word.append(word)
                    generated_word_prob.append(word_p)
                word = token
                word_p = p

        generated_word.append(word)
        generated_word_prob.append(word_p)

        ## seperator token post_process
        if mustbeprocessed:
            generated_word_proc = []
            generated_word_prob_proc = []
            BREAK = False

            for i, (word, word_p) in enumerate(zip(generated_word, generated_word_prob)):
                if (word in self.week_tokens) and (generated_word_proc[-1] in ['[EOHW]', '[EOW]']):
                    sep = generated_word_proc.pop()
                    sep_p = generated_word_prob_proc.pop()
                    word = sep+word          # token은 이어 붙이고
                    word_p = sep_p*word_p    # beam P는 곱하고

                generated_word_proc.append(word)
                generated_word_prob_proc.append(word_p)

                if BREAK:
                    return generated_word_proc, generated_word_prob_proc
            return [generated_word_proc, generated_word_prob_proc]

        else:
            return [generated_word, generated_word_prob]



    def word_to_label(
            self,
            sample_in_batch,
    ) -> list:

        # result
        generated_word_to_pred_mccb = []

        for generated_word in sample_in_batch:
            # word내 '#'제거
            generated_word = [word.replace('#', '') for word in generated_word]

            # word가 mrch이므로 mccb로 바꿈
            pred_mccb = list(map(lambda x: self.mrch_to_mccb[x], generated_word))

            # mccb가 []인 것은 제외 (=가맹점에 mapping되는 업종)
            pred_mccb = list(filter(lambda x: len(x), pred_mccb))

            # 예측한 mccb중에서 중복된 mccb를 제거
            # 어차피 발생했는지 여부만 중요함(빈도 x)
            # e.g. n일 동안 ['mccb1', 'mccb1', 'mccb1']이든, ['mccb1']이든 label은 1
            pred_mccb_uniq = list(set(pred_mccb))
            del pred_mccb

            # {mccb:class}가 mapping되지 않는 것 filtering
            filtered_target_mccb = list(
                filter(lambda x: x is not None,
                       [self.target_mccb.get(mccb_pred) for mccb_pred in pred_mccb_uniq])
            )

            generated_word_to_pred_mccb.append(filtered_target_mccb)

        return generated_word_to_pred_mccb


    def uniq_list(
            self,
            L:list,
    ) -> list:
        '''
        nested list에서 uniq element를 골라내기 위한 함수
        (Set operation이 hashable item에서만 사용이 가능함 --> list: unhashable)

        Y_batch에 같은 class가 있을 경우 이를 중복 제거하는 함수

        Parameters:
            L: Y_batch가 통째로 들어감. dim = (batch_size, top_k)

        Examples:
            # batch_size = 2, top_k = 10인 경우,
            >>> L = [[7], [7], [], [1, 7], [], [7], [], [], [1], []]
            >>> result = [[], [1], [7], [1, 7]]
        '''

        tuple_L = [list(map(tuple, element)) for element in L]
        return [list(map(list, set(element))) for element in tuple_L]




    def MultiLabelBinarizer(
            self,
            y_batch:list,
            class_num:int = 9,
            is_y_true:bool = False,
    ) -> np.array:
        '''
        class번호만 있는 y를 multilabel화시키는 함수

        Parameters:
            y_batch : sample별 class집합이 batch_size만큼 있는 형태(개별 요소는 여러 개의 class를 가질 수 있음)
                dim = (batch_size, top_k에서 중복제거 된 차원-각기 다름)
            is_y_true : y_true인 경우 size가 다르기 때문에 (batch_size, 1) -> 다른 처리를 해 주어야 함

        Examples:
            # batch_size = 4, class_num = 9인 경우,
            # 아래와 같은 형태를 multilabel화 시킴
            >>> y_batch = [[2], [1, 3], [9], [3, 5, 7]]
            >>> return = [
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 1, 0, 1, 0, 0],
            ]
        '''

        ml = [[0]*(class_num+1) for _ in range(len(y_batch))]

        for i, topk in enumerate(y_batch):
            if not is_y_true:
                for pred in topk:
                    if len(pred): # pred != []
                        list(map(m[i].__setitem__, pred, [1]*len(pred))) # operation만 수행
            else:
                list(map(m[i].__setitem__, topk, [1] * len(topk)))

        return np.asarray(ml)


    def check_similarity_ROUGE_BLEU(
            self,
            generated_texts:list,
            answer:str
    ) -> (list, list, list, list, str):
        '''
        Parameters:
            generated_texts : top_k의 각각 token들을 word단위로 이은 것. dim = (top_k, num_words)
            answer : answer string
        '''

        if len(answer) < 2:
            answer = '</>'

        from rouge import Rouge
        rouge_12l = Rouge()

        # http://stackoverflow.com/questions/51748890/
        def str_intersaction(str1, str2):
            comm = []
            for char in set(str1):
                comm.extend(char * min(str1.count(char), str2.count(char)))
            return comm

        n_inters_list = [] # select longest match
        for i, word in enumerate(generated_texts):
            string_to_tokens = self.tokenizer.tokenize(word)

            if not set(string_to_tokens).difference(set(self.seperator+self.week_tokens)):
                word = '</>'
                generated_texts[i] = word

            n_inters_tmp = len(str_intersaction(answer, word.replace(" ". ""))) / len(answer)
            n_inters_list.append(n_inters_tmp)


        BLEU_score_list = []
        ROUGE_score_list = []
        for generated_text in generated_texts:
            # if len(generated_text) < 2:
            #     generated_text = '</>'
            BLEU_score_custom = modified_precision(generated_text.replace('#', '').replace('][', '] ['),
                                                   [answer.replace('][', '] [')])
            BLEU_score = BLEU_score_custom
            ROUGE_score = rouge_12l.get_scores(generated_text.replace('#', '').replace('][', '] ['),
                                               answer.replace('][', '] ['))

            BLEU_score_list.append(BLEU_score)
            ROUGE_score_list.append(ROUGE_score[0]['rouge-l']['r']) # rouge-l의 recall

        return n_inters_list, BLEU_score_list, ROUGE_score_list, generated_texts, answer




    def averager(
            self,
            metric:Union[Metric],
            decimal_point=4
    ) -> float:
        '''
        결과 저장 dictionary에 쌓인 metric들을 한 번에 평균내는 함수
        각 metric마다 평균 내는 방법이 다르므로 이를 try-except구문으로 처리

        Parameters:
            metric : metric.Metric인지 혹은 keras.metrics.AUC 중 하나의 type
            decimal_point : 반올림(=round) 자리 수
        '''
        try:
            return round(float(metric.compute_average()), decimal_point)
        except:
            return round(float(metric.result().numpy()), decimal_point)



    ##### main evaluation #####
    @torch.no_grad():
    def do_eval(
            self,
            batch_size:int,
            to_test_df:pd.DataFrame,
            top_k:int,
            rank:int,
            log_interval:int = 100,
    ):
        self.model = self.model.half()
        self.model = self.model.to(rank)

        N = len(to_test_df)
        steps = (N // batch_size)+1

        self.load_mapping_dictionary()
        self.preprocess_dictionary()

        self.accuracy = Metric('accuracy')
        self.bleu = Metric('BLEU')
        self.rouge = Metric('Rouge')

        # metric_dict
        # multi label의 각 label마다 track하는 지표들 dictionary
        metrics_per_class = defaultdict(lambda: [Metric('precision'), Metric('recall'), Metric('accuary'), Metric('f1')])

        # useHeader = True # for textfile out
        useHeader = False # resume 시

        for i in tqdm(range(0, N, batch_size), desc=f'Rank #{rank}: Eval test examples', position=rank, leave=True):
            batch = to_test_df[i:i+batch_size]

            # to_test_df의 column참조
            csnos = batch.csno
            ys = batch.Y
            y_txt = batch.Y_txt

            encodings = self.tokenizer(batch.X.to_list(), return_tensors='pt',
                                      padding=True, truncation=True,
                                      max_length=self.model.config.max_position_embedding-self.max_new_tokens,
                                      add_special_tokens=False)

            encodings = BatchEncoding({k: encodings[k].to(rank) for k in encodings})


            with torch.no_grad():
                outputs = self.model.generate(**encodings,
                                              bad_words_ids = self.bad_word_ids,
                                              max_new_tokens = self.max_new_tokens,
                                              min_new_tokens = self.min_new_tokens,
                                              num_beams = top_k * self.num_beams,
                                              num_beam_groups = top_k,
                                              diversity_penalty = float(top_k),
                                              num_return_sequences = top_k,
                                              output_scores = True,
                                              return_dict_in_generate = True,
                                              eos_token_ids = self.tokenizer.convert_ids_to_tokens(self.week_tokens),
                                              torch_dtype = self.model.config.torch_dtype)

                # outputs.scores = Shape(num_return_sequences, (num_beams, vocab_size))
                # shape : (batch_size * num_return_sequences, num_generated)
                transition_scores = self.compute_transition_scores(
                    outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
                ).reshape(batch_size, top_k, -1)


                output_length = encodings.input_ids.shape[1] + (transition_scores.cpu().numpy() < 0)
                transition_probs = torch.exp(transition_scores.cpu() / output_length)
                # torch.Size([16, num_return_sequences, 9])
                del transition_scores
                torch.cuda.empty_cache()

                generated_ids = outputs.sequences[:, encodings.input_ids.shape[1]:].reshape(batch_size, top_k, -1)
                del outputs, encodings
                torch.cuda.empty_cache()

                # token -> word, transition_score(2/)
                # generated_ids = (batch_size, top_k, generated_token_num)
                # generated_id = (top_k, generated_token_num)
                processed_output = [list(map(self.token_to_word_with_transition_prob, generated_id, transition_prob))
                    for generated_id, transition_prob in zip(generated_ids, transition_probs)
                ]
                processed_output = np.array(processed_output, dtype='O') # for object type in pd.DataFrame
                del generated_ids
                torch.cuda.empty_cache()

                generated_word_batch, generated_word_prob_batch = processed_output[:,:,0], processed_output[:,:,1]
                del processed_output
                torch.cuda.empty_cache()
                gc.collect()
                ## 분기 시점

                ###########################################
                ###### track (1) : evaluation metric ######
                ###########################################

                # word -> 업종 label로 바꾸기
                # 단위 : 개별 sample
                for generated_words in generated_word_batch:
                    generated_word_to_pred_mccb = []

                    for generated_word in generated_words:
                        generated_word = [word.replace('#', '') for word in generated_word]
                        pred_mccb = list(map(lambda x: self.mrch_to_mccb[x], generated_word))
                        pred_mccb = list(filter(lambda x: len(x), pred_mccb))
                        pred_mccb_uniq = list(set(pred_mccb))

                        filtered_target_mccb = list(
                            filter(lambda x: x is not None, [self.target_mccb.get(mccb_pred) for mccb_pred in pred_mccb_uniq])
                        )
                        generated_word_to_pred_mccb.append(filtered_target_mccb)
                        del filtered_target_mccb
                batched_pred_topk = list(map(self.word_to_label, generated_word_batch))


                # multi-label화 --- BERT와 비교하기 위해
                y_pred_ml = self.MultiLabelBinarizer(self.uniq_list(batched_pred_topk))
                y_true_ml = self.MultiLabelBinarizer(batch.Y.to_list(), is_y_true=True)
                del batched_pred_topk
                torch.cuda.empty_cache()


                # update metric dict
                for c in range(10):
                    report = classification_report(y_true_ml[:, c], y_pred_ml[:, c],
                                                   labels=[0, 1], digits=4, output_dict=True,
                                                   zero_division=0)

                    metrics_per_class[c][0].update([report['l']['precision']])
                    metrics_per_class[c][1].update([report['l']['recall']])
                    metrics_per_class[c][2].update([accuracy_score(y_true_ml[:, c], y_pred_ml[:, c])]) # pred, label이 모두 0일 경우 report에 안뜸
                    metrics_per_class[c][3].update([report['l']['f1-score']])


                # for display
                metrics_per_class_for_print = {
                    k: map(lambda x: self.averager(x), v) for k, v in metrics_per_class.item()
                }

                metric_df = pd.DataFrame(metrics_per_class_for_print)
                metric_df.columns = ['class_'+str(c) for c in metric_df.columns]
                metric_df.index = ['Precision', 'Recall', 'Accuracy', 'F1_score']


                ###########################################
                ##### track (2) : text generation 측정 #####
                ###########################################

                # 단위 : 개별 sample
                if ((i // batch_size) % log_interval) == 0:
                    print_switch = True
                else:
                    print_switch = False


                batch_n_inters = []
                batch_BLEUs = []
                batch_ROUGEs = []
                batch_y_gen = []
                batch_y_gen_true = []

                for csno, y_gen_topk, y_gen_topk_prob, y_true in zip(csnos, generated_word_batch,\
                                                                     generated_word_prob_batch, batch.Y_txt):
                    y_gen_topk_txt = list(map(lambda x: ' '.join(x), y_gen_topk))

                    y_gen_topk_prob = [
                        list(
                            map(lambda x: np.round(x.numpy(), 4), topk_p_word)
                        )
                        for topk_p_word in y_gen_topk_prob
                    ]

                    # compute metrics by the current batch
                    n_inters, BLEU, ROUGE, X_correction, Y_correction = self.check_similarity_ROUGE_BLEU(y_gen_topk_txt,
                                                                                                         y_true)

                    batch_n_inters.append(n_inters)
                    batch_BLEUs.append(BLEU)
                    batch_ROUGEs.append(ROUGE)
                    batch_y_gen.append(X_correction)
                    batch_y_gen_true.append(Y_correction)

                    # # log print
                    # if print_switch:
                    #     display(pd.DataFrame({
                    #         'csno': [csno] + ['']*(len(y_gen_topk)-1),
                    #         'y_gen': X_correction,
                    #         'y_gen_beam_prob': y_gen_topk_prob,
                    #         'n_inter': list(map(lambda x: round(x, 3), n_inters)),
                    #         'bleu': list(map(lambda x: round(x, 3), BLEU)),
                    #         'rouge': list(map(lambda x: round(x, 3), ROUGE)),
                    #         'y_true': [Y_correction] + ['']*(len(y_gen_topk)-1),
                    #     }).set_index('csno'))

            # metric update
            self.accuracy.update(batch_n_inters)
            self.bleu.update(batch_BLEUs)
            self.rouge.update(batch_ROUGEs)

            if print_switch:
                print(f"{(i//batch_size)+1}/{steps}] | Process {rank} \
                | Top {top_k} n_inters : {self.accuracy.compute_average():>.4f} \
                | Top_{top_k} BLEU : {self.bleu.compute_average():>.4f} \
                | Top_{top_k} ROUGE : {self.rouge.compute_average():>.4f}")


            del generated_word_batch, generated_word_prob_batch
            torch.cuda.empty_cache()
            gc.collect()



            # 1. 필요한 컬럼들
            #   1-1. sequence_generation 평가 -> csno, pred_seq_mrch, pred_seq_mrch_prob, scores
            #   1-2. vs BERT 평가 -> csno, pred_label_mccb, y_label, metrics -> 적재?
            # 2. Time stamp? -> log로 돌리기
            # 3. 계산할 것들 -> 중간중간 print? (hive table? or local csv file)

            vsBERT_batch_output_df = pd.DataFrame({
                'csno' : csnos,
                'y_pred_ml' : y_pred_ml.tolist(),
                'y_pred_txt': batch_y_gen,
                'y_true_ml' : y_true_ml.tolist(),
                'y_true_txt': batch_y_gen_true
            })
            vsBERT_batch_output_df['part_dt'] = self.today

            torch.cuda.empty_cache()
            gc.collect()

            ### 결과 저장 (1) hive table 적재
            if self.save_type == 'hive':
                with Hyspark(**self.hyspark_param) as hs:
                    sai_packer.register(hs)
                    hc, sc, ss = hs.hive_context, hs.spark_context, hs.spark_session
                    check_hive_available(hc)

                    output_pyspark_df_schema = StructType([
                        StructField("csno", StringType()),
                        StructField("y_pred_ml", ArrayType(IntegerType()),
                        StructField("y_pred_txt", ArrayType(StringType()),
                        StructField("y_true_ml", ArrayType(IntegerType()),
                        StructField("y_true_txt", ArrayType(StringType()),
                        StructField("part_dt", StringType())
                    ])

                    vsBERT_batch_output_pyspark_df = ss.createDataFrame(vsBERT_batch_output_df, output_pyspark_df_schema)
                    vsBERT_batch_output_pyspark_df.write.saveAsTable('hivedb.table_name',
                                                                     mode='append', partitionBy='part_dt')
                del vsBERT_batch_output_pyspark_df



            ### 결과 저장 (2) csv file 저장
            else:
                vsBERT_batch_output_df.to_csv('file_name.csv'
                                              sep='\t',
                                              encoding='utf-8',
                                              index=False,
                                              mode='a',
                                              header=useHeader)
                useHeader = False

            del batch, csnos, ys, y_txt
            del vsBERT_batch_output_df
            torch.cuda.empty_cache()
            gc.collect()


        del metrics_per_class
















