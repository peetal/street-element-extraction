�	Ϡ��_�@Ϡ��_�@!Ϡ��_�@      ��!       "h
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails'Ϡ��_�@i���1@1�F�&ڑ@IK�|%��/@r0*	x�&1� �@2c
,Iterator::Root::Prefetch::ParallelMapV2::Map���֗T@!��Vݑ�X@)Y��#�T@1����X@:Preprocessing2l
5Iterator::Root::Prefetch::ParallelMapV2::Map::BatchV2LOX�e�?!N�TFQ
�?)��ɍ"k�?1�j� �?:Preprocessing2z
BIterator::Root::Prefetch::ParallelMapV2::Map::BatchV2::TensorSlice�����?!������?)����?1������?:Preprocessing2O
Iterator::Root::Prefetch�=Ab��?!��#�Ѡ?)�=Ab��?1��#�Ѡ?:Preprocessing2E
Iterator::Root%�����?!ج�|Nx�?)Xt�5=(�?1�F�,IM�?:Preprocessing2^
'Iterator::Root::Prefetch::ParallelMapV2C�B�Y��?!������?)C�B�Y��?1������?:Preprocessing:�
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
�Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
�Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
�Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
�Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)�
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis�
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*noI���E�@Q;�o�5JX@Zno#You may skip the rest of this page.B�
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown�
	i���1@i���1@!i���1@      ��!       "	�F�&ڑ@�F�&ڑ@!�F�&ڑ@*      ��!       2      ��!       :	K�|%��/@K�|%��/@!K�|%��/@B      ��!       J      ��!       R      ��!       Z      ��!       b      ��!       JGPUb q���E�@y;�o�5JX@�
"t
Vtf_bert_for_token_classification/bert/encoder/layer_._11/output/dense/Tensordot/MatMulMatMul��|?!��|?0"t
Vtf_bert_for_token_classification/bert/encoder/layer_._10/output/dense/Tensordot/MatMulMatMul���AK�{?!���� ��?0"s
Utf_bert_for_token_classification/bert/encoder/layer_._7/output/dense/Tensordot/MatMulMatMul��s��{?!DT�P��?0"s
Utf_bert_for_token_classification/bert/encoder/layer_._8/output/dense/Tensordot/MatMulMatMulp��D��{?!�5���ߛ?0"s
Utf_bert_for_token_classification/bert/encoder/layer_._2/output/dense/Tensordot/MatMulMatMul���װ{?!��4�e�?0"s
Utf_bert_for_token_classification/bert/encoder/layer_._0/output/dense/Tensordot/MatMulMatMuld:-k{?!� a�LӤ?0"�
jgradient_tape/tf_bert_for_token_classification/bert/encoder/layer_._8/output/dense/Tensordot/MatMul/MatMulMatMulL����{?!�P �J7�?0"�
kgradient_tape/tf_bert_for_token_classification/bert/encoder/layer_._10/output/dense/Tensordot/MatMul/MatMulMatMul`(���{?!���ʩ��?0"�
jgradient_tape/tf_bert_for_token_classification/bert/encoder/layer_._6/output/dense/Tensordot/MatMul/MatMulMatMul�UJ��z?!�K*����?0"�
jgradient_tape/tf_bert_for_token_classification/bert/encoder/layer_._7/output/dense/Tensordot/MatMul/MatMulMatMul�*�$A�z?!k�]\E+�?0Q      Y@Y������@aVUUUU�W@qcF�Ex�?y����È;?"�	
device�Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*�
�<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2�
=type.googleapis.com/tensorflow.profiler.GenericRecommendation�
nono*�Only 0.0% of device computation is 16 bit. So you might want to replace more 32-bit Ops by 16-bit Ops to improve performance (if the reduced accuracy is acceptable).2no:
Refer to the TF2 Profiler FAQ2"Nvidia GPU (Kepler)(: B 