import { Mult } from "./metatype"

export type MLOprand = any

export type Tensor<N extends number, C extends number, H extends number, W extends number> = [N, C, H, W]

export type Filter<O extends number, I extends number, H extends number, W extends number> = [O, I, H, W]

export type width = 180

export type height = 120

export type upsampledWidth = Mult<width, 4>

export type upsampledHeight = Mult<height, 4>

export type pool1Width = Mult<width, 2>

export type pool1Height = Mult<height, 2>

export type pool2Width = width

export type pool2Height = height

export type frameCount = 6

export type kernelSize = 3

export type state = {
    context: any,
    builder: any,
    graph: any,
    frameCount: frameCount,
    width: number,
    height: number,
    upsampledWidth: number,
    upsampledHeight: number,
    weightForZeroUpsamplingAllFeatures: Filter<12, 12, 4, 4>,
    weightForZeroUpsamplingCurrentFrame: Filter<4, 4, 4, 4>,
    input_view: Tensor<frameCount, 3, height, width>,
    input_depth: Tensor<frameCount, 1, height, width>,
    all_rgbd: Tensor<frameCount, 4, height, width>,
    all_features: Tensor<frameCount, 12, height, width>,
    all_features_upsampled: Tensor<frameCount, 12, upsampledHeight, upsampledWidth>,
    arr_last_features_reweighted: Array<Tensor<1, 12, upsampledHeight, upsampledWidth>>,
    output: Tensor<1, 3, upsampledHeight, upsampledHeight>
}