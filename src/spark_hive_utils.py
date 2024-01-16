from hyspark import Hyspark # Spark TF에서 만든 사내 spark.
from contextlib import contextmanager, redirect_stdout
from pyarrow import cpu_count, set_cpu_count
from IPython.utils.io import capture_output
from pyspark.sql.types import NumericType
from time import perf_counter
from subprocess import Popen, PIPE
from uuid import uuid4
from warnings import warn as warn_
from os import getcwd as getcwd_
from os import remove as remove_
from os import mkdir as mkdir_
from shutil import rmtree
from six import string_types
from io import StringIO

import pandas as pd
import numpy as np
import signal
import sys

# 시간 출력 함수 (용례: with elapsed_time(...))
'''
Parameters:
  format_string : 하나의 '%f' Symbol이 있어야 함
  verbose : False일 경우, 시간은 측정하지만 출력은 하지 않음 
'''
@contextmanager
def elapsed_time(format_string='Elapsed time: %f seconds', verbose=True):
  start_time = perf_counter()
  yield
  elapsed_time = perf_counter() - start_time
  if verbose:
    print(format_string % elapsed_time)




# Pyspark에서 Hive로 데이터 접근이 가능 지 체크하는 함수
'''
Parameters:
  hive_context : Pyspark SQLcontext (Hyspark의 리턴값에서 .hive_context)
                (None일 경우, Pyspark를 새로 초기화하여 Hive에 연결해보고, 자동 종료함)
  timeout : Hive에 접속을 시도하고 몇 초 동안 기다릴 지 설정 (기본 60초)
  verbose : True의 경우, 시간 내 접속이 잘 되면 현재 시각을, 안되면 경고를 출력함
'''
def check_hive_available(hive_context=None, timeout=60, verbose=False):
  # 내부 함수 : 거듭 사용을 위해 함수로 만듦
  def _inner_check_hive(hive_context):
    try:
      with time_limit(timeout):
        df = hive_context.sql('SELECT CURRENT_TIMESTAMP')
        current_time = df.first()[0]
        if verbose:
          print('Current Time: %s' % current_time)
        return True
    except:
        warn_('''Error establishing Hive connection.
                 Contact your system administrator.
                 (Unknown problem when communicating with Thrift server.)''')

  if hive_context is None:
    with create_spark_session(verbose=verbose) as hs:
      return _inner_check_hive(hs.hive_context)
  else:
    return _inner_check_hive(hive_context)


# Hyspark 사용을 Context Manager하도록 도와주는 함수
'''
Parameters:
  verbose : True일 경우, Hyspark가 출력하는 모든 출력문을 그대로 출력함
  check_hive_connection : True일 경우, Pyspark에서 Hive가 잘 접근되는지 체크함
                          (체크 후, 문제가 있을 시 경고 메시지 출력)
  enable_dynamic_partition : True일 경우, Hive에서 Dynamic Partition을 사용함
                            기본적으로 Hive에서는 Dynamic Partition을 Disable로 해두기 떄문에 
                            default는 False로 지정함
  optimize_num_shuffle_partitions : 'spark.sql.shuffle.partitions' 속성의 설정값을 
                                    최적화할 지 여부를 결정 (True이면 optimize_shuffle_partitions 함수 실행)
  multifplier : optimize_shuffle_partitions 함수의 인자
  **hyspark_kwargs:  
    - app_name : Spark Cluster에 등록될 작업명 (식별 가능한 문자열을 지정)
    - mem_per_core : 8까지는 무난하게 올릴 수 있으나 8이 초과되면 Core수 할당이 줄어듦 ('genenral'일 경우 11까지도 가능)
    - instance : 자원 할당량 조절에 관여함, 기본은 'mini'로 설정되어 있음
Usage:
  with create_spark_session(args) as hs:
    hc = hs.hive_context
    ss = hs.spark_context
    ss = hs.spark_session
'''
@contextmanager
def create_spark_session(app_name=None, mem_per_core=2, verbose=False,
                         check_hive_connection=True,
                         enable_dynamic_partition=False,
                         optimize_num_shuffle_partitions=False, multiplier=3,
                         **hyspark_kwargs):
  app_name = app_name or 'SPARK-%s' % uuid4()

  try:
    # (출력 redirection) 
    # Hyspark의 Standard Ouptut은 buf로, IPython Output은 ipython_captured로 모음
    buf = StringIO()
    with capture_output() as ipython_captured, redirect_stdout(buf):
      hs = HySpark(app_name, mem_per_core, **hyspark_kwargs)

    # Pyspark에서 Hive접근이 원활한 지 체크함 (안되면 경고 메세지 출력)
    if check_hive_connection:
      check_hive_available(hs.hive_context, verbose=verbose)

    # verbose=True일 때, Hyspark의 구현 순서 그대로 출력함
    if verbose:
      if 'ipykernel' in sys.modules:
        for o in ipython_captured.outputs:
          display(o)
      print(buf.getvalue())

    # 이 Session에서 수행할 작업 중, 데이터에 Partition을 부여하여 Insert할 경우,
    # Dynamic Partitioning(명시적이 아닌 Partition 컬럼 값에 의해 데이터 분할 결정)이 필요하다면
    # enable_dynamic_partition=True로 하여 아래와 같이 설정을 바꾸어 줌
    if enable_dynamic_partition:
      conf1 = ('hive.exec.dynamic.partition', 'true')
      conf2 = ('hive.exec.dynamic.partition.mode', 'nonstrict')
      hs.hive_context_setConf(*conf1)
      hs.hive_context_setConf(*conf2)
      if verbose:
        print("Set '%s'='%s' %conf1")
        print("Set '%s'='%s' %conf2")

    # 'spark.sql.shuffle.partitions' 속성의 설정값을 최적화하는 함수 실행
    # 아래 optimize_shuffle_partitions 함수 설명 참조
    if optimize_num_shuffle_partitions:
      optimize_shuffle_partitions(hs.hive_context, hs.spark_context,
                                  multiplier, verbose)

    yield hs
  finally:
    hs.stop()   # 어떤 경우에도 반드시 자원 반환이 실행되어야 함(Error시에도)
              

      
