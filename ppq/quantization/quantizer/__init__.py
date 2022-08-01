from .AcademicQuantizer import (ACADEMIC_INT4_Quantizer,
                                ACADEMIC_Mix_Quantizer, ACADEMICQuantizer)
from .base import BaseQuantizer
from .DSPQuantizer import PPL_DSP_Quantizer, PPL_DSP_TI_Quantizer
from .MetaxQuantizer import MetaxChannelwiseQuantizer, MetaxTensorwiseQuantizer
from .MyQuantizer import ExtQuantizer
from .NXPQuantizer import NXP_Quantizer
from .ORTQuantizer import ORT_PerChannelQuantizer, ORT_PerTensorQuantizer
from .PPLQuantizer import (PPLCUDA_INT4_Quantizer,
                           PPLCUDAMixPrecisionQuantizer, PPLCUDAQuantizer)
from .TRTQuantizer import TensorRTQuantizer
from .FPGAQuantizer import FPGAQuantizer
from .NCNNQuantizer import NCNNQuantizer
from .OpenvinoQuantizer import OpenvinoQuantizer
from .TengineQuantizer import TengineQuantizer