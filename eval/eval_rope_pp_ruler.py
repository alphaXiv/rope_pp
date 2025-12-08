from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import RoPEPPCausalLM

with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k
    from opencompass.configs.datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_datasets_8k
    from opencompass.configs.datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_datasets_16k
    from opencompass.configs.datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_datasets_32k
    from opencompass.configs.datasets.ruler.ruler_64k_gen import ruler_datasets as ruler_datasets_64k

datasets = []
datasets += ruler_datasets_4k
datasets += ruler_datasets_8k
datasets += ruler_datasets_16k
datasets += ruler_datasets_32k
datasets += ruler_datasets_64k

models = [

    ('RoPE-DCLM-376M-32k', 'SII-xrliu/RoPE-DCLM-376M-32k', {'imag': False, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EH-DCLM-376M-32k', 'SII-xrliu/RoPEPP_EH-DCLM-376M-32k', {'imag': True, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EC-DCLM-376M-32k', 'SII-xrliu/RoPEPP_EC-DCLM-376M-32k', {'imag': True, 'imag_mode': 'imag2'}, 64), 

    ('RoPE-DCLM-776M-32k', 'SII-xrliu/RoPE-DCLM-776M-32k', {'imag': False, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EH-DCLM-776M-32k', 'SII-xrliu/RoPEPP_EH-DCLM-776M-32k', {'imag': True, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EC-DCLM-776M-32k', 'SII-xrliu/RoPEPP_EC-DCLM-776M-32k', {'imag': True, 'imag_mode': 'imag2'}, 64), 

    ('RoPE-DCLM-1_5B-32k', 'SII-xrliu/RoPE-DCLM-1_5B-32k', {'imag': False, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EH-DCLM-1_5B-32k', 'SII-xrliu/RoPEPP_EH-DCLM-1_5B-32k', {'imag': True, 'imag_mode': 'imag1'}, 64), 
    ('RoPEPP_EC-DCLM-1_5B-32k', 'SII-xrliu/RoPEPP_EC-DCLM-1_5B-32k', {'imag': True, 'imag_mode': 'imag2'}, 64), 

]

models = [
    dict(
        type=RoPEPPCausalLM, abbr=abbr, rope_config=rope_config, path=path, 
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1, 
        run_cfg=dict(num_gpus=1, num_procs=1),
    ) for abbr, rope_config, ckpt, max_out_len in models
]

work_dir = './outputs_xrliu/rope_pp_ruler/'

infer = dict(
    partitioner=dict(type=NaivePartitioner), 
    runner=dict(
        type=LocalRunner,
        task=dict(type=OpenICLInferTask), 
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=64, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)
