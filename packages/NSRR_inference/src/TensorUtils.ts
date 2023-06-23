import { sizeOfShape } from "./common/utils"
import { Tensor } from "./type"

export let buildTensorWithValue = <N extends number, C extends number, H extends number, W extends number>(builder, dimensions: [N, C, H, W], value: number): Tensor<N, C, H, W> => {
    return builder.constant({ type: 'float32', dimensions: dimensions }, new Float32Array(sizeOfShape(dimensions)).fill(value)
    )
}