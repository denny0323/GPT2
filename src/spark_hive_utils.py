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
              
# 현 Spark 환경에서 Total Executor Core의 갯수를 return
# Total Executor Core의 갯수가 Parallelism의 동시 Task 실행 수를 결정함
# (즉, 한번에 동시에 실행되는 Task의 개수라고 생각할 수 있음)
def get_total_num_executor_cores(spark_context):
  n_instances = get_spark_conf(spark_context, 'spark.executor.instances')
  n_cores_per_executor = get_spark_conf(spark_context, 'spark.executor.cores')
  return int(n_instances) * int(n_cores_per_executor)

# Spark Context환경설정 Literal(conf_str)에 대한 설정값을 str로 리턴
# (예: get_conf(sc, 'spark.driver.memory')는 해당 설정값을 문자열로 리턴함
def get_spark_conf(spark_context, conf_str):
  return spark_context._conf.get(conf_str)

# Spark SQLContext관련 설정값을 리턴
# 예를 들어, 'spark.sql.shffle.partitions', 'spark.sql.files.maxPartitionBytes' 등의
# 환경설정명에 대한 값을 얻어올 수 있음 (위 함수와 혼동 주의)
def get_sql_conf(hive_context, conf_str):
  return hive_context.getConf(conf_str)

# Spark SQLContext관련 설정값을 바꿈
# 예를 들어, 'spark.sql.shffle.partitions', 'spark.sql.files.maxPartitionBytes' 등의
# 환경설정을 바꿀 수 있음
def set_sql_conf(hive_context, conf_str, value):
  return hive_context.setConf(conf_str, value)



# 'spark.sql.shuffle.partitions' 속성의 설정값을 최적화하는 함수
''' 
속성 : Configures the number of partitions to use
           when shuffling data for joins or aggregations. (Default 200)) 
Description : Hyspark에서 기본 Driver를 사용하면 이 때는 Total Executor Cores 수가 
              65로 잡히므로 적절하지만 Orge Driver 등을 사용하면 165까지도 잡히기 때문에,
              이 값을 적절히 2~4배 정도로 불려줄 때, 데이터의 Join 및 Attribute의 병렬 연산에
              더 도움이 될 것. 그대로 200일 경우 200개의 Task 중 165개가 Assign되어 돌아간 후
              나머지 35개가 돌아갈 것. 만약 330으로 2배 개수로 잡았다면, 시간이 더 절약됨
'''
# Spark session을 잡은 후, 바로 다음에 실행하면, 이후 작업들이 이 설정 기준으로 실행됨
def optimize_shuffle_partitions(hive_context, spark_context, multiplier=3, 
                                verbose=False):
  property_str = 'spark.sql.shuffle.partitions'
  n_cores = get_total_num_executor_cores(spark_context)
  new_val = n_cores * multiplier

  if verbose:
    prev_val = int(get_seq_conf(hive_context, property_str))
    print("Current Value of '%s': '%d' -> Change to %d" %
          (property_str, prev_val, new_val))

  return set_sql_conf(hive_context, property_str, new_val)



# 'spark.sql.shuffle.partitions'속성의 설정값을 바꾸는 함수 (의미는 위 함수 참조)
def change_shuffle_partitions(hive_context, num_shuffle_partitions):
  return set_sql_conf(hive_context, 'spark.sql.shuffle.partitions',
                      num_shuffle_partitions)

# Pandas DataFrame의 Memory 사용량을 MegaBytes로 리턴
def get_mem_usage_in_megabytes(pandas_df):
  return float(sum(pandas_df.memory_usage())) / 1024 / 1024


# sql_as_pandas_w_pyspark, df_as_pandas_with_pyspark에서 사용되는 함수
# cast dict에 대한 Validation 및 처리
def _check_cast_dict(cast_dict):
  if cast_dict is None:
    return {}
    
  if not isinstance(cast_dict, dict):
    raise ValueError('cast_dict must be a dict.')

  # 지원되는 Cast String은 Pyspark 2.4.0 기준을 따름 ('short' 추가)
  supported_cast_strings = ('int', 'tinyint', 'smallint', 'short', 'biging',
                            'boolean', 'float', 'double', 'string')
  diff = set(cast_dict.values()) - set(supported_cast_strings)
  if len(diff) > 0:
    raise ValueError("cast_dict contains unsupported values: %s. "
                     "Supported values are %s. " %
                     (diff, supported_cast_strings))
  return cast_dict


# sql_as_pandas_with_pyspark, df_as_pandas_with_pyspark에서 사용되는 함수
# Pyspark DataFrame에 대해, convert_decimal, cast_dict에 따른 type conversion logic이 적용된
# String Expression을 리턴하는 함수 (selectExpr에 사용됨)
def _get_cast_expr(pyspark_df, convert_decimal, cast_dict,
                   numeric_to_float, double_to_float, bigint_in_int,
                   verbose=False, verbose_cast=20):
  cast_str_list = []
  cast_info_list = []
  cast_template = 'CAST(%s AS %s) AS %s'

  default_real = 'float' if double_to_float else 'double'
  default_int = 'int' if bigint_to_int else 'bigint'

  for s in pyspark_df.schema:
    dtype = s.dataType
    dname = dtype.typeName()

    # cast_dict에 casting 규칙이 있으면 이를 우선함
    # casting 규칙이 없는데 convert_decimal=True인 경우에는 실수부의 유무로 판별
    if s.name in cast_dict:
      type_str = cast_dict[s.name]
    elif numeric_to_float and isinstance(dtype, NumericType):
      type_str = 'float'
    elif convert_decimal and dname.startswith('decimal'):
      type_str = default_int if dtype.scale == 0 else default_real
    elif double_to_float and dname == 'double':
      type_str = default_real
    elif bigint_to_int and dname == 'bigint':
      type_str = default_int
    else:
      cast_str_list.append(s.name)
      continue

    cast_str_list.append(cast_template %(s.name, type_str, s.name))
    cast_info_list.append('%s AS %s' %(s.name, type_str))

  num_cast = len(cast_info_list)
  if verbose and num_cast > 0:
    if verbose_cast > 0 and verbose_cast < num_cast:
      cast_info_str = ', '.join(cast_info_list[:verbose]) + ' ...'
    else:
      cast_info_str = ', '.join(cast_info_list)
    print('Type Conversion (%d Columns): %s' %(num_cast, cast_info_str))

  # cast_info_list가 비어 있으면 Type Casting 없음의 의미로 빈 리스트 return
  return [] if num_cast == 0 else cast_str_list
