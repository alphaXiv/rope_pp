from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import MaskCausalLM

with read_base():
    from opencompass.configs.datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_datasets_4k

datasets = []
datasets += ruler_datasets_4k


models = [

    # 376m-rope_pp_ec-short

    ('RoPEPP_EC-DCLM-376M-4k', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, None, 64),

    # ('RoPEPP_EC-DCLM-376M-4k-both_0_2', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 0.2}, 64),
    # ('RoPEPP_EC-DCLM-376M-4k-both_0_4', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 0.4}, 64),
    # ('RoPEPP_EC-DCLM-376M-4k-both_0_6', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 0.6}, 64),
    # ('RoPEPP_EC-DCLM-376M-4k-both_0_8', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 0.8}, 64),
    # ('RoPEPP_EC-DCLM-376M-4k-both_1_0', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 1.0}, 64),
    # ('RoPEPP_EC-DCLM-376M-4k-both_1_2', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 1.2}, 64),
    # ('RoPEPP_EC-DCLM-376M-4k-both_1_5', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 1.5}, 64),

    ('RoPEPP_EC-DCLM-376M-4k-real_0_2', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.2}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-real_0_4', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.4}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-real_0_6', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.6}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-real_0_8', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.8}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-real_1_0', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.0}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-real_1_2', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.2}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-real_1_5', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.5}, 64),

    ('RoPEPP_EC-DCLM-376M-4k-imag_0_2', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.2}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-imag_0_4', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.4}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-imag_0_6', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.6}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-imag_0_8', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.8}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-imag_1_0', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.0}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-imag_1_2', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.2}, 64),
    ('RoPEPP_EC-DCLM-376M-4k-imag_1_5', 'SII-xrliu/RoPEPP_EC-DCLM-376M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.5}, 64),

    # 776m-rope_pp_ec-short

    ('RoPEPP_EC-DCLM-776M-4k', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, None, 64),

    # ('RoPEPP_EC-DCLM-776M-4k-both_0_2', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 0.2}, 64),
    # ('RoPEPP_EC-DCLM-776M-4k-both_0_4', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 0.4}, 64),
    # ('RoPEPP_EC-DCLM-776M-4k-both_0_6', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 0.6}, 64),
    # ('RoPEPP_EC-DCLM-776M-4k-both_0_8', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 0.8}, 64),
    # ('RoPEPP_EC-DCLM-776M-4k-both_1_0', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 1.0}, 64),
    # ('RoPEPP_EC-DCLM-776M-4k-both_1_2', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 1.2}, 64),
    # ('RoPEPP_EC-DCLM-776M-4k-both_1_5', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'both', 'noise_std': 1.5}, 64),

    ('RoPEPP_EC-DCLM-776M-4k-real_0_2', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.2}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-real_0_4', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.4}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-real_0_6', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.6}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-real_0_8', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 0.8}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-real_1_0', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.0}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-real_1_2', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.2}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-real_1_5', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'real', 'noise_std': 1.5}, 64),

    ('RoPEPP_EC-DCLM-776M-4k-imag_0_2', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.2}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-imag_0_4', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.4}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-imag_0_6', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.6}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-imag_0_8', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 0.8}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-imag_1_0', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.0}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-imag_1_2', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.2}, 64),
    ('RoPEPP_EC-DCLM-776M-4k-imag_1_5', 'SII-xrliu/RoPEPP_EC-DCLM-776M-4k', {'imag': True, 'imag_mode': 'imag2'}, {'noise_target': 'imag', 'noise_std': 1.5}, 64),

]

models = [
    dict(
        type=MaskCausalLM, abbr=abbr, path=path, 
        rope_config=rope_config, mask_config=mask_config, 
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, batch_size=1, 
        run_cfg=dict(num_gpus=1, num_procs=1),
    ) for abbr, path, rope_config, mask_config, max_out_len in models
]


work_dir = './outputs_xrliu/rope_pp_ruler-mask/'

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
        max_num_workers=96, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)
