from .common import train
import os
from omegaconf import OmegaConf
from collections import OrderedDict
from detectron2.config import LazyCall as L
from data.build import MultiTaskDataLoader
from modeling.meta_arch.multitask_v3 import MultiTaskBatchFuse

# segmentation
from data.transforms.seg_transforms import Normalize, Resize, ResizeByShort
from data.build_segmentation import build_segmentation_trainloader, \
    build_segementation_test_dataset
from evaluation.segmentation_evaluator import SegEvaluatorInfer

# classification
from data.build import build_reid_test_loader_lazy
from data.transforms.build import build_transforms_lazy

from data.build_cls import build_hierachical_test_set
from evaluation.common_cls_evaluator import CommonClasEvaluatorSingleTaskInfer

# detection
from data.build_trafficsign import build_cocodet_set, build_cocodet_loader_lazy
from evaluation.cocodet_evaluator import CocoDetEvaluatorSingleTaskInfer

dataloader=OmegaConf.create()
_root = os.getenv("FASTREID_DATASETS", "datasets")

seg_num_classes=19

# NOTE
# trian/eval模式用于构建对应的train/eval Dataset, 需提供样本及标签;
# infer模式用于构建InferDataset, 只需提供测试数据, 最终生成结果文件用于提交评测, 在训练时可将该部分代码注释减少不必要评测

dataloader.test = [
    
    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(
            segmentation=L(build_segmentation_trainloader)(
                data_set=L(build_segementation_test_dataset)(
                        dataset_name="InferDataset",
                        dataset_root=_root + '/test_data/seg/', 
                        transforms=[
                            L(Resize)(
                            ),
                            L(Normalize)(
                                mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]
                        )],
                        mode='test',
                        is_padding=True),
                total_batch_size=3, 
                worker_num=3, 
                drop_last=False, 
                shuffle=False,
                num_classes=seg_num_classes,
                is_train=False,
            ),
        ),
    ),

    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(
            fgvc=L(build_reid_test_loader_lazy)(
                test_set=L(build_hierachical_test_set)(
                    dataset_name = "FGVCInferDataset",
                    test_dataset_dir = _root + '/test_data/cls/test/',
                    transforms=L(build_transforms_lazy)(
                        is_train=False,
                        size_test=[448, 448],
                        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
                    ),
                    is_train=False,  # infer mode
                    
                ),
                test_batch_size=3,
                num_workers=3,
            ),
        ),
    ),

    L(MultiTaskDataLoader)(
        cfg=dict(sample_mode='batch',),
        task_loaders=L(OrderedDict)(           
            trafficsign=L(build_cocodet_loader_lazy)(
                data_set=L(build_cocodet_set)(
                    is_padding=True,
                    dataset_name="COCOInferDataSet",
                    transforms=[
                        dict(Decode=dict(),),
                        dict(Resize=dict(
                            target_size=[1280, 1280], 
                            keep_ratio=False)
                            ),
                        # dict(MultiscaleTestResize=dict(
                        #     origin_target_size=[2048, 2048],
                        #     target_size=[1024, 1024],
                        #     use_flip=True
                        # )),
                        dict(NormalizeImage=dict(
                            is_scale=True, 
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
                            ),
                        dict(Permute=dict()),
                    ],
                    image_dir='test.txt',
                    anno_path='val.json',  # 可以采用验证集或训练集json,只是为了获取类别ID映射关系
                    dataset_dir= _root + '/test_data/dec/',
                    data_fields=['image', 'im_id', 'im_file'],
                ),
                total_batch_size=2,
                num_workers=2,
                batch_transforms=[
                    dict(PadMaskBatch=dict(pad_to_stride=32, return_pad_mask=False),),
                ],
                is_train=False,
                shuffle=False, 
                drop_last=False, 
                collate_batch=False,
            ),
        ),    
    ),
]

# NOTE
# trian/eval模式用于eval;
# infer模式则用于生成测试集预测结果(可直接提交评测), 在训练时可注释该部分代码减少不必要评测

dataloader.evaluator = [
    # segmentation
    L(SegEvaluatorInfer)(
    ),  # infer mode

    # classification
    L(CommonClasEvaluatorSingleTaskInfer)(
        cfg=dict(), num_classes=196
    ),   # infer mode

    # detection
    L(CocoDetEvaluatorSingleTaskInfer)(
        classwise=False, 
        output_eval=None,     
        bias=0, 
        IouType='bbox', 
        save_prediction_only=False,
        parallel_evaluator=True,
        num_valid_samples=3067, 
    ),  # infer mode

]



from ppdet.modeling import ShapeSpec
from modeling.backbones.swin_transformer import SwinTransformer_small_patch4_window7_224_maskformer
from modeling.heads.swin_cls_head import SwinClsHead

from modeling.heads.detr import DETR
from ppdet.modeling.transformers.dino_transformer import DINOTransformer
from ppdet.modeling.transformers.matchers import HungarianMatcher
from ppdet.modeling.heads.detr_head import DINOHead
from ppdet.modeling.post_process import DETRBBoxPostProcess
from ppdet.modeling.losses.detr_loss import DINOLoss
from modeling.heads.maskformer_head import MaskFormer
from modeling.losses.maskformer_loss import MaskFormerLoss

backbone=L(SwinTransformer_small_patch4_window7_224_maskformer)(
)

trafficsign_num_classes=45
use_focal_loss=True

model=L(MultiTaskBatchFuse)(
    backbone=backbone,
    pretrained=False,
    pretrain_path='pretrained/SwinTransformer_tiny_patch4_window7_224_pretrained.pdparams',
    heads=L(OrderedDict)(
        segmentation=L(MaskFormer)(
            num_classes=seg_num_classes,
            loss=L(MaskFormerLoss)(
                num_classes=seg_num_classes,
                eos_coef=0.1,
            ),
        ),

        fgvc=L(SwinClsHead)(
            embedding_size=384, 
            class_num=196,
        ),

        trafficsign=L(DETR)(
            transformer=L(DINOTransformer)(
                            num_classes=trafficsign_num_classes,
                            hidden_dim=256,
                            num_queries=900,
                            position_embed_type='sine',
                            return_intermediate_dec=True,
                            backbone_feat_channels=[96, 192, 384, 768],
                            num_levels=4,
                            num_encoder_points=4,
                            num_decoder_points=4,
                            nhead=8,
                            num_encoder_layers=6,
                            num_decoder_layers=6,
                            dim_feedforward=2048,
                            dropout=0.0,
                            activation="relu",
                            num_denoising=100,
                            label_noise_ratio=0.5,
                            box_noise_scale=1.0,
                            learnt_init_query=True,
                            eps=1e-2),
            detr_head=L(DINOHead)(loss=L(DINOLoss)(
                            num_classes=trafficsign_num_classes,
                            loss_coeff={"class": 1, "bbox": 5, "giou": 2, "no_object": 0.1, "mask": 1, "dice": 1},
                            aux_loss=True,
                            use_focal_loss=use_focal_loss,
                            matcher=L(HungarianMatcher)(
                                matcher_coeff={"class": 2, "bbox": 5, "giou": 2},
                                use_focal_loss=use_focal_loss,)   
                            )
           ),
            post_process=L(DETRBBoxPostProcess)(
                            num_classes=trafficsign_num_classes,
                            num_top_queries=50,
                            use_focal_loss=use_focal_loss,
                            ),
        ),
    ),
    pixel_mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
    pixel_std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
)


train.amp.enabled = False
train.init_checkpoint = 'averaged_model_weights_fade_aug.pdparams'

train.output_dir = 'outputs/test_swin_small_230730_1'
