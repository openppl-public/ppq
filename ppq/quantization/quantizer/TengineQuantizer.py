from typing import Union

import torch
from ppq.core import (TensorQuantizationConfig,OperationQuantizationConfig, QuantizationPolicy,
                      QuantizationProperty, QuantizationStates, RoundingPolicy,
                      TargetPlatform)

from ppq.IR import BaseGraph, GraphCommandProcessor, Operation


from .base import BaseQuantizer


class TengineQuantizer(BaseQuantizer):
    def __init__(
        self, graph: Union[BaseGraph, GraphCommandProcessor]
    ) -> Union[torch.Tensor, list, dict]:
        super().__init__(graph=graph)
        self._num_of_bits = 8
        self._quant_min = 0
        self._quant_max = 256


    def init_quantize_config(
        self, operation: Operation) -> OperationQuantizationConfig:
        base_quant_config = self.create_default_quant_config(
            policy=self.quantize_policy, rounding=self.rounding_policy,
            operation_meta=operation.meta_data, num_of_bits=self._num_of_bits,
            quant_max=self._quant_max, quant_min=self._quant_min,
            observer_algorithm='percentile'
        )

        if operation.type in {'Conv', 'Gemm'}:
            assert operation.num_of_input > 0, 'Seems you got a Computing layer with no parameters.'

            if operation.type == 'Conv':
                conv_weight_config = base_quant_config.input_quantization_config[1]
                conv_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.ASYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_TENSOR
                )
                base_quant_config.input_quantization_config[1] = \
                    TensorQuantizationConfig.convert_from_tensor_config(
                        convert_from = conv_weight_config,
                        offsets = None, scales  = None, channel_axis = 0
                    )
                base_quant_config.input_quantization_config[1].observer_algorithm = 'Minmax'

                group        = operation.attributes.get('group', 1)
                dilations    = operation.attributes.get('dilations', [1, 1])
                strides      = operation.attributes.get('strides', [1, 1])
                kernel_shape = operation.attributes.get('kernel_shape')
                if group == 1 and all([i == 1 for i in dilations]) and all([j == 1 for j in strides])\
                    and all([k == 3 for k in kernel_shape]):
                    base_quant_config.input_quantization_config[1].num_of_bits = 6
                    base_quant_config.input_quantization_config[1].quant_max   = +31
                    base_quant_config.input_quantization_config[1].quant_min   = -31

            elif operation.type == 'Gemm':
                assert operation.attributes.get('transB', 0) and operation.attributes.get('alpha', 1.0) == 1.0 \
                    and operation.attributes.get('beta', 1.0) == 1.0
                gemm_weight_config = base_quant_config.input_quantization_config[1]
                gemm_weight_config.policy = QuantizationPolicy(
                    QuantizationProperty.ASYMMETRICAL +
                    QuantizationProperty.LINEAR +
                    QuantizationProperty.PER_TENSOR
                )
                base_quant_config.input_quantization_config[1] = \
                    TensorQuantizationConfig.convert_from_tensor_config(
                        convert_from = gemm_weight_config,
                        offsets = None, scales  = None, channel_axis = 0
                    )
                base_quant_config.input_quantization_config[1].observer_algorithm = 'Minmax'

            # if operation has bias
            if operation.num_of_input > 2:
                bias_config = base_quant_config.input_quantization_config[-1]
                bias_config.state = QuantizationStates.FP32

            base_quant_config.output_quantization_config[0].state = QuantizationStates.FP32
        return base_quant_config

    @ property
    def target_platform(self) -> TargetPlatform:
        return TargetPlatform.TENGINE_INT8

    @ property
    def default_platform(self) -> TargetPlatform:
        return TargetPlatform.TENGINE

    @ property
    def quant_operation_types(self) -> set:
        return {
            'Conv', 'GlobalAveragePool', 'AveragePool',
            'Relu', 'Add', 'Mul', 'Clip', 'Sigmoid',
            'MatMul', 'Gemm', 'Concat', 'LeakyRelu'}
    @ property
    def quantize_policy(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            QuantizationProperty.ASYMMETRICAL +
            QuantizationProperty.LINEAR +
            QuantizationProperty.PER_TENSOR
        )

    @ property
    def rounding_policy(self) -> RoundingPolicy:
        return RoundingPolicy.ROUND_HALF_EVEN

    @ property
    def activation_fusion_types(self) -> set:
        return {'Relu', 'Clip'}
