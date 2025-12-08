from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import PythiaCausalLM

with read_base():

    from opencompass.configs.datasets.truthfulqa.truthfulqa_gen_5ddc62 import truthfulqa_datasets

    from opencompass.configs.datasets.piqa.piqa_ppl_1cf9f0 import piqa_datasets
    from opencompass.configs.datasets.hellaswag.hellaswag_ppl_47bff9 import hellaswag_datasets
    from opencompass.configs.datasets.winogrande.winogrande_ppl_55a66e import winogrande_datasets

    from opencompass.configs.datasets.ARC_e.ARC_e_ppl_a450bd import ARC_e_datasets

    from opencompass.configs.datasets.gpqa.gpqa_ppl_6bf57a import gpqa_datasets
    from opencompass.configs.datasets.siqa.siqa_ppl_ced5f6 import siqa_datasets
    from opencompass.configs.datasets.obqa.obqa_ppl_c7c154 import obqa_datasets

    from opencompass.configs.datasets.SuperGLUE_AX_b.SuperGLUE_AX_b_ppl import AX_b_datasets
    from opencompass.configs.datasets.SuperGLUE_AX_g.SuperGLUE_AX_g_ppl import AX_g_datasets
    from opencompass.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_ppl import BoolQ_datasets
    from opencompass.configs.datasets.SuperGLUE_CB.SuperGLUE_CB_ppl import CB_datasets
    from opencompass.configs.datasets.SuperGLUE_COPA.SuperGLUE_COPA_ppl import COPA_datasets
    from opencompass.configs.datasets.SuperGLUE_MultiRC.SuperGLUE_MultiRC_ppl import MultiRC_datasets
    from opencompass.configs.datasets.SuperGLUE_ReCoRD.SuperGLUE_ReCoRD_gen import ReCoRD_datasets
    from opencompass.configs.datasets.SuperGLUE_RTE.SuperGLUE_RTE_ppl import RTE_datasets
    from opencompass.configs.datasets.SuperGLUE_WiC.SuperGLUE_WiC_ppl import WiC_datasets
    from opencompass.configs.datasets.SuperGLUE_WSC.SuperGLUE_WSC_ppl import WSC_datasets

datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

models = [

    ('Pythia-DCLM-376M-4k', 'SII-xrliu/Pythia-DCLM-376M-4k', {}, 64), 
    ('Pythia-DCLM-776M-4k', 'SII-xrliu/Pythia-DCLM-776M-4k', {}, 64), 

]

models = [
    dict(
        type=PythiaCausalLM, abbr=abbr, rope_config=rope_config, path=path, 
        model_kwargs={'flash_attention': True}, max_out_len=max_out_len, max_seq_len=2048, 
        batch_size=16, run_cfg=dict(num_gpus=1),
    ) for abbr, path, rope_config, max_out_len in models
]

work_dir = './outputs/rope_pp_short/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=4000, gen_task_coef=15),
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
