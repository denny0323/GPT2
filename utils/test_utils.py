import numpy as np
from collections import Counter
from nltk import ngrams

def simple_count(tokens, n):
  return Counter (ngrams(tokens, n))

def count_clip(candidate, reference_list, n=1):
  # Ca 문장에서 n-gram 카운트
  ca_cnt = simple_count(candidate, n)
  max_ref_cnt_dict = dict()

  for ref in reference_list: 
    # Ref 문장에서 n-gram 카운트
    ref_cnt = simple_count(ref, n)

    # 각 Ref 문장에 대해서 비교하여 n-gram의 최대 등장 횟수를 계산.
    for n_gram in ref_cnt: 
      if n_gram in max_ref_cnt_dict:
        max_ref_cnt_dict[n_gram] = max(ref_cnt[n_gram], max_ref_cnt_dict[n_gram])
      else:
        max_ref_cnt_dict[n_gram] = ref_cnt[n_gram]

  return {
        # count_clip = min(count, max_ref_count)
        n_gram: min(ca_cnt.get(n_gram, 0), max_ref_cnt_dict.get(n_gram, 0)) for n_gram in ca_cnt
     }

def modified_precision(candidate, reference_list, n=1):
  clip_cnt = count_clip(candidate, reference_list, n) 
  total_clip_cnt = sum(clip_cnt.values()) # 분자

  cnt = simple_count(candidate, n)
  total_cnt = sum(cnt.values()) # 분모

  # 분모가 0이 되는 것을 방지
  if total_cnt == 0: 
    total_cnt = 1

  # 분자 : count_clip의 합, 분모 : 단순 count의 합 ==> 보정된 정밀도
  return (total_clip_cnt / total_cnt)


def sentence_modified_precision(candidate_list, reference):
  return [modified_precision(candidate, [reference]) for candidate in candidate_list]


