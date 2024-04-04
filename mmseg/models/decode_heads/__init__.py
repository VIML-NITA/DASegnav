from .fcn_head import FCNHead
from .ours_head_class_attn import OursHeadClassAtt
from .psp_head import PSPHead
from .segnav_head import APFormerHead,  APFormerHead2, APFormerHeadSingle, APFormerHeadMulti, APFormerHead2_rebuttal
from .cc_head import CCHead

__all__ = [
    'FCNHead', 'OursHeadClassAtt', 'PSPHead', 'APFormerHead',  'CCHead', 'APFormerHead2', 'APFormerHeadSingle', 'APFormerHeadMulti', 'APFormerHead2_rebuttal'
]
# , 'APFormerHead2', 'APFormerHeadSingle', 'APFormerHeadMulti', 'APFormerHead2_rebuttal'