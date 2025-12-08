from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import RoPEPPCausalLM

with read_base():

    from opencompass.configs.datasets.babilong.babilong_2k_gen import babiLong_2k_datasets
    from opencompass.configs.datasets.babilong.babilong_4k_gen import babiLong_4k_datasets
    from opencompass.configs.datasets.babilong.babilong_8k_gen import babiLong_8k_datasets
    from opencompass.configs.datasets.babilong.babilong_16k_gen import babiLong_16k_datasets
    from opencompass.configs.datasets.babilong.babilong_32k_gen import babiLong_32k_datasets
    from opencompass.configs.datasets.babilong.babilong_64k_gen import babiLong_64k_datasets

datasets = []

datasets += babiLong_2k_datasets
datasets += babiLong_4k_datasets
datasets += babiLong_8k_datasets
datasets += babiLong_16k_datasets
datasets += babiLong_32k_datasets
datasets += babiLong_64k_datasets

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

work_dir = './outputs_xrliu/rope_pp_babilong/'

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
