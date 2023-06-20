// TODO change to abstract type
export type MLOprand = any

export type Tensor<n, c, h, w> = MLOprand

export type Filter<o, i, h, w> = MLOprand

export type width = 180

export type height = 120

export type upsampledWidth = 720

export type upsampledHeight = 480

export type pool1Width = 360

export type pool1Height = 240

export type pool2Width = 180

export type pool2Height = 120

export type state = {
    context: any,
    builder: any,
    graph: any,
    input_view: Tensor<6, 3, height, width>,
    input_depth: Tensor<6, 1, height, width>,
    all_features: Tensor<6, 12, height, width>,
    all_features_upsampled: Tensor<6, 12, upsampledHeight, upsampledWidth>,
    arr_last_features_reweighted: Array<Tensor<1, 12, upsampledHeight, upsampledWidth>>,
    output: Tensor<1,3,upsampledHeight, upsampledHeight>
}