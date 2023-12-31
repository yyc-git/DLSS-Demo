import '@webmachinelearning/webnn-polyfill'
import { range } from './common/Array'
import { buildConstantByNpy, sizeOfShape } from "./common/utils"
import { height, Filter, Tensor, width, state, upsampledHeight, upsampledWidth, pool1Width, pool1Height, pool2Height, pool2Width, kernelSize } from './type'
import { Mult } from "./metatype"
import { buildTensorWithValue } from './TensorUtils'
// import { getDimensions } from './TensorUtils'

//create state

export let createState = ([width, height]): state => {
    return {
        context: null,
        builder: null,
        graph: null,
        frameCount: 6,
        width: width,
        height: height,
        upsampledWidth: width * 4,
        upsampledHeight: height * 4,
        weightForZeroUpsamplingAllFeatures: null,
        weightForZeroUpsamplingCurrentFrame: null,
        input_view: null,
        input_depth: null,
        all_rgbd: null,
        all_features: null,
        all_features_upsampled: null,
        arr_last_features_reweighted: null,
        output: null
    }
}

export let _prepareWeightForZeroUpsampling = <C extends number>(state: state, c: C): Filter<typeof c, typeof c, 4, 4> => {
    // refer to: https://github.com/pytorch/pytorch/issues/7911#issuecomment-392835113

    let builder = state.builder

    let dimension = [1, 1, 4, 4]
    let value0 = new Float32Array(sizeOfShape(dimension)).fill(0.0)
    let value1 = new Float32Array(sizeOfShape(dimension)).fill(0.0)
    value1[0] = 1.0

    let weightArr: Array<Filter<1, typeof c, 4, 4>> = range(0, c - 1).map(i => {
        let arr = range(0, c - 1).reduce((result, j) => {
            if (i === j) {
                result[j] = builder.constant(
                    { type: 'float32', dimensions: dimension },
                    value1
                )
            }
            else {
                result[j] = builder.constant(
                    { type: 'float32', dimensions: dimension },
                    value0
                )
            }

            return result
        }, [])

        return builder.concat(arr, 1)
    })

    let weight: Filter<typeof c, typeof c, 4, 4> = builder.concat(weightArr, 0)

    return weight
}

export let init = async (state, contextOptions) => {
    let context = await (navigator as any).ml.createContext(contextOptions)

    let tf = context.tf
    //TODO really use webgpu? or just webgl?
    // await tf.setBackend("webgpu")
    // await tf.setBackend("webgl")
    tf.env().set('WEBGL_EXP_CONV', true);
    await tf.setBackend("webgl")
    await tf.ready()

    let builder = new MLGraphBuilder(context)

    state = {
        ...state,
        context,
        builder,
    }

    return {
        ...state,
        weightForZeroUpsamplingAllFeatures: _prepareWeightForZeroUpsampling(state, 12),
        weightForZeroUpsamplingCurrentFrame: _prepareWeightForZeroUpsampling(state, 4),
    }

}

export let createComputeGraphOfInput = (state) => {
    let { builder, width, height } = state

    let input_viewShape = [1, 3, height, width]
    let input_view = builder.concat(
        [
            builder.input('input_view1', { type: 'float32', dimensions: input_viewShape }),
            builder.input('input_view2', { type: 'float32', dimensions: input_viewShape }),
            builder.input('input_view3', { type: 'float32', dimensions: input_viewShape }),
            builder.input('input_view4', { type: 'float32', dimensions: input_viewShape }),
            builder.input('input_view5', { type: 'float32', dimensions: input_viewShape }),
            builder.input('input_view6', { type: 'float32', dimensions: input_viewShape }),
        ],
        0
    )

    let input_depthShape = [1, 1, height, width]
    let input_depth = builder.concat(
        [
            builder.input('input_depth1', { type: 'float32', dimensions: input_depthShape }),
            builder.input('input_depth2', { type: 'float32', dimensions: input_depthShape }),
            builder.input('input_depth3', { type: 'float32', dimensions: input_depthShape }),
            builder.input('input_depth4', { type: 'float32', dimensions: input_depthShape }),
            builder.input('input_depth5', { type: 'float32', dimensions: input_depthShape }),
            builder.input('input_depth6', { type: 'float32', dimensions: input_depthShape }),
        ],
        0
    )

    return {
        ...state,
        input_view,
        input_depth,
    }
}

let _getPadding = () => [1, 1, 1, 1]

let _getStrides = () => [1, 1]

let _buildFeatureExtractModel = (builder, input_rgbd: Tensor<1, 4, height, width>, [conv1Weight, conv2Weight, conv3Weight]: [
    Filter<32, 4, kernelSize, kernelSize>,
    Filter<32, 32, kernelSize, kernelSize>,
    Filter<8, 32, kernelSize, kernelSize>,
], [conv1Bias, conv2Bias, conv3Bias]: [
    Tensor<1, 1, 1, 32>,
    Tensor<1, 1, 1, 32>,
    Tensor<1, 1, 1, 8>,
]): Tensor<1, 8, height, width> => {
    let conv1 = builder.conv2d(
        input_rgbd,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv1Bias
        }
    )
    let conv2 = builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv2Bias
        }
    )
    let conv3 = builder.conv2d(
        conv2,
        conv3Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv3Bias
        }
    )

    return conv3
}

export let createComputeGraphOfFeatureExtract = (state: state, weights, biases): state => {
    let {
        builder,
        width,
        height,
        input_view,
        input_depth
    } = state

    let input_all_rgbd: Tensor<6, 4, height, width> = builder.concat([input_view, input_depth], 1)

    let all_featuresWithoutInput = builder.concat(
        range(0, 6 - 1).map(i => {
            return _buildFeatureExtractModel(builder, builder.slice(input_all_rgbd, [i, 0, 0, 0], [1, 4, height, width]), weights[i], biases[i])
        }),
        0
    )

    let all_features = builder.concat([input_all_rgbd, all_featuresWithoutInput], 1)

    return {
        ...state,
        all_rgbd: input_all_rgbd,
        all_features
    }
}

export let _zeroUpsampling = <N extends number, C extends number, H extends number, W extends number>(builder, tensor: Tensor<N, C, H, W>, weightForZeroUpsampling, n: N, c: C, h: H, w: W): Tensor<N, C, Mult<H, 4>, Mult<W, 4>> => {
    // refer to: https://github.com/pytorch/pytorch/issues/7911#issuecomment-392835113

    return builder.concat(
        range(0, n - 1).map(i => {
            // console.log(n, tensor)
            return builder.convTranspose2d(
                builder.slice(tensor, [i, 0, 0, 0], [1, c, h, w]),
                weightForZeroUpsampling,
                {
                    strides: [4, 4],
                }
            )
        }),
        0
    )
}

export let createComputeGraphOfZeroUpsampling = (state: state): state => {
    let {
        builder,
        all_features
    } = state

    return {
        ...state,
        all_features_upsampled: _zeroUpsampling(builder, all_features,
            state.weightForZeroUpsamplingAllFeatures,
            state.frameCount, 12, state.height, state.width)
    }
}

export let _remap = <N extends number, C extends number, H extends number, W extends number>(builder, x: Tensor<N, C, H, W>, xDimensions: [N, C, H, W], in_range: [number, number], out_range: [number, number]) => {
    return builder.div(
        builder.mul(
            builder.add(x,
                buildTensorWithValue(builder, xDimensions, - in_range[0])
            ),
            buildTensorWithValue(builder, xDimensions, out_range[1] - out_range[0])
        ),
        buildTensorWithValue(builder, xDimensions, (in_range[1] - in_range[0]) + out_range[0])
    )
}

let _multiplyWeightMap = (state: state, builder, weighting_map: Tensor<1, 5, upsampledHeight, upsampledWidth>, input_last_frame_features_upsampled: Tensor<5, 12, upsampledHeight, upsampledWidth>): Array<Tensor<1, 12, upsampledHeight, upsampledWidth>> => {
    return range(0, 5 - 1).reduce((arr, i) => {
        let input_last_frame_feature_upsampled: Tensor<1, 12, upsampledHeight, upsampledWidth> = builder.slice(input_last_frame_features_upsampled, [i, 0, 0, 0], [1, 12, state.upsampledHeight, state.upsampledWidth])

        let weighting_one_last_frame_map: Tensor<1, 1, upsampledHeight, upsampledWidth> = builder.slice(weighting_map, [0, i, 0, 0], [1, 1, state.upsampledHeight, state.upsampledWidth])

        let tmp: Array<Tensor<1, 1, upsampledHeight, upsampledWidth>> = range(0, 12 - 1).reduce((tmp, j) => {
            tmp.push(
                builder.mul(
                    builder.slice(input_last_frame_feature_upsampled, [0, j, 0, 0], [1, 1, state.upsampledHeight, state.upsampledWidth]),
                    weighting_one_last_frame_map
                )
            )

            return tmp
        }, [])

        arr.push(builder.concat(tmp, 1))

        return arr
    }, [])
}

let _buildFeatureReweightingModel = (state: state, builder,
    input_current_frame_upsampled: Tensor<1, 4, upsampledHeight, upsampledWidth>,
    input_last_frame_features_upsampled: Tensor<5, 12, upsampledHeight, upsampledWidth>,
    [conv1Weight, conv2Weight, conv3Weight]: [
        Filter<40, 24, kernelSize, kernelSize>,
        Filter<40, 40, kernelSize, kernelSize>,
        Filter<5, 40, kernelSize, kernelSize>,
    ],
    [conv1Bias, conv2Bias, conv3Bias]: [
        Tensor<1, 1, 1, 40>,
        Tensor<1, 1, 1, 40>,
        Tensor<1, 1, 1, 5>,
    ]
): Array<Tensor<1, 12, upsampledHeight, upsampledWidth>> => {
    let reweight_feed_in: Tensor<1, 24, upsampledHeight, upsampledWidth> = range(0, 5 - 1).reduce((reweight_feed_in, i) => {
        // let input_last_frame_feature_upsampled = builder.slice(input_last_frame_features_upsampled, [i, 0, 0, 0], [1, 12, state.upsampledHeight, state.upsampledWidth])

        // return builder.concat(
        //     [
        //         reweight_feed_in,
        //         builder.slice(input_last_frame_feature_upsampled, [0, 0, 0, 0], [1, 4, state.upsampledHeight, state.upsampledWidth])
        //     ],
        //     1
        // )

        return builder.concat(
            [
                reweight_feed_in,
                builder.slice(input_last_frame_features_upsampled, [i, 0, 0, 0], [1, 4, state.upsampledHeight, state.upsampledWidth])
            ],
            1
        )
    }, input_current_frame_upsampled)

    let conv1 = builder.conv2d(
        reweight_feed_in,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv1Bias
        }
    )
    let conv2 = builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv2Bias
        }
    )
    let conv3 = builder.conv2d(
        conv2,
        conv3Weight,
        {
            activation: builder.tanh(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv3Bias
        }
    )

    let weighting_map: Tensor<1, 5, upsampledHeight, upsampledWidth> = _remap(builder, conv3, [1, 5, state.upsampledHeight, state.upsampledWidth], [-1, 1], [0, 10])

    return _multiplyWeightMap(state, builder, weighting_map, input_last_frame_features_upsampled)
}

export let createComputeGraphOfFeatureReweighting = (state: state, weights, biases): state => {
    let {
        builder,
        all_rgbd,
        all_features_upsampled
    } = state

    let input_current_frame_upsampled: Tensor<1, 4, upsampledHeight, upsampledWidth> = _zeroUpsampling(builder, all_rgbd,
        state.weightForZeroUpsamplingCurrentFrame,
        1, 4, state.height, state.width) as any

    let input_last_frame_features_upsampled = builder.slice(all_features_upsampled, [1, 0, 0, 0], [5, 12, state.upsampledHeight, state.upsampledWidth])

    let arr_last_features_reweighted = _buildFeatureReweightingModel(state, builder,
        input_current_frame_upsampled,
        input_last_frame_features_upsampled,
        weights,
        biases
    )

    return {
        ...state,
        arr_last_features_reweighted: arr_last_features_reweighted
    }
}

let _builderEncoder1 = (builder, input: Tensor<1, 72, upsampledHeight, upsampledWidth>,
    [conv1Weight, conv2Weight]: [Filter<64, 72, kernelSize, kernelSize>,
        Filter<32, 64, kernelSize, kernelSize>
    ],
    [conv1Bias, conv2Bias]: [
        Tensor<1, 1, 1, 64>,
        Tensor<1, 1, 1, 32>,
    ]
): Tensor<1, 32, upsampledHeight, upsampledWidth> => {
    let conv1 = builder.conv2d(
        input,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv1Bias
        }
    )

    return builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv2Bias
        }
    )
}

let _builderEncoder2 = (builder, input: Tensor<1, 32, pool1Height, pool1Width>, [conv1Weight, conv2Weight]: [Filter<64, 32, kernelSize, kernelSize>,
    Filter<64, 64, kernelSize, kernelSize>
],
    [conv1Bias, conv2Bias]: [
        Tensor<1, 1, 1, 64>,
        Tensor<1, 1, 1, 64>,
    ]
): Tensor<1, 64, pool1Height, pool1Width> => {
    let conv1 = builder.conv2d(
        input,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv1Bias
        }
    )

    return builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv2Bias
        }
    )
}

let _builderCenter = (builder, input: Tensor<1, 64, pool2Height, pool2Width>, [conv1Weight, conv2Weight]: [Filter<128, 64, kernelSize, kernelSize>,
    Filter<128, 128, kernelSize, kernelSize>
],
    [conv1Bias, conv2Bias]: [
        Tensor<1, 1, 1, 128>,
        Tensor<1, 1, 1, 128>,
    ]
): Tensor<1, 128, pool1Height, pool1Width> => {
    let conv1 = builder.conv2d(
        input,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv1Bias
        }
    )

    let conv2 = builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv2Bias
        }
    )

    // refer to https://github.com/webmachinelearning/webnn/issues/358
    return builder.resample2d(conv2, {
        mode: "linear",
        scales: [2, 2],
        axes: [2, 3]
    })
}

let _builderCat1 = (builder, input: Tensor<1, 192, pool1Height, pool1Width>, convWeight: Filter<128, 192, kernelSize, kernelSize>,
    convBias: Tensor<1, 1, 1, 128>
): Tensor<1, 128, pool1Height, pool1Width> => {
    return builder.conv2d(
        input,
        convWeight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: convBias
        }
    )
}

let _builderDecoder2 = (builder, input: Tensor<1, 128, pool1Height, pool1Width>, [conv1Weight, conv2Weight]: [Filter<64, 128, kernelSize, kernelSize>,
    Filter<64, 64, kernelSize, kernelSize>
],
    [conv1Bias, conv2Bias]: [
        Tensor<1, 1, 1, 64>,
        Tensor<1, 1, 1, 64>,
    ]
): Tensor<1, 64, upsampledHeight, upsampledWidth> => {
    let conv1 = builder.conv2d(
        input,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv1Bias
        }
    )

    let conv2 = builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: conv2Bias
        }
    )

    return builder.resample2d(conv2, {
        mode: "linear",
        scales: [2, 2],
        axes: [2, 3]
    })
}

let _builderCat2 = (builder, input: Tensor<1, 96, upsampledHeight, upsampledWidth>, convWeight: Filter<32, 96, kernelSize, kernelSize>,
    convBias: Tensor<1, 1, 1, 32>
): Tensor<1, 32, upsampledHeight, upsampledWidth> => {
    return builder.conv2d(
        input,
        convWeight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: convBias
        }
    )
}

let _builderDecoder1 = (builder, input: Tensor<1, 32, upsampledHeight, upsampledWidth>, convWeight: Filter<3, 32, kernelSize, kernelSize>,
    convBias: Tensor<1, 1, 1, 3>
): Tensor<1, 3, upsampledHeight, upsampledWidth> => {
    return builder.conv2d(
        input,
        convWeight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides(),
            bias: convBias
        }
    )
}

let _buildFeatureReconstructionModel = (builder,
    input_current_frame_feature_upsampled: Tensor<1, 12, upsampledHeight, upsampledWidth>,
    input_arr_last_frame_features_reweighted: Array<Tensor<1, 12, upsampledHeight, upsampledWidth>>,
    [
        encoder1Weight,
        encoder2Weight,
        centerWeight,
        cat1Weight,
        decoder2Weight,
        cat2Weight,
        decoder1Weight
    ],
    [
        encoder1Bias,
        encoder2Bias,
        centerBias,
        cat1Bias,
        decoder2Bias,
        cat2Bias,
        decoder1Bias
    ]
): Tensor<1, 3, upsampledHeight, upsampledWidth> => {
    let x: Tensor<1, 72, upsampledHeight, upsampledWidth> =
        input_arr_last_frame_features_reweighted.reduce((result, input_last_frame_features_reweighted) => {
            return builder.concat(
                [
                    result,
                    input_last_frame_features_reweighted
                ], 1
            )
        }, input_current_frame_feature_upsampled)

    let x_encoder_1 = _builderEncoder1(builder, x, encoder1Weight, encoder1Bias)
    let x_encoder_1_pool = builder.maxPool2d(x_encoder_1, {
        windowDimensions: [2, 2],
        strides: [2, 2]
    })
    let x_encoder_2 = _builderEncoder2(builder, x_encoder_1_pool, encoder2Weight, encoder2Bias)
    let x_encoder_2_pool = builder.maxPool2d(x_encoder_2, {
        windowDimensions: [2, 2],
        strides: [2, 2]
    })
    let x_center = _builderCenter(builder, x_encoder_2_pool, centerWeight, centerBias)

    let x_cat_1 = _builderCat1(builder, builder.concat([x_center, x_encoder_2], 1), cat1Weight, cat1Bias)
    let x_decoder_2 = _builderDecoder2(builder, x_cat_1, decoder2Weight, decoder2Bias)

    let x_cat_2 = _builderCat2(builder, builder.concat([x_decoder_2, x_encoder_1], 1), cat2Weight, cat2Bias)
    return _builderDecoder1(builder, x_cat_2, decoder1Weight, decoder1Bias)
}

export let createComputeGraphOfReconstruction = (state: state, weights, biases): state => {
    let {
        builder,
        all_features_upsampled,
        arr_last_features_reweighted
    } = state

    return {
        ...state,
        output: _buildFeatureReconstructionModel(builder,
            builder.slice(all_features_upsampled, [0, 0, 0, 0], [1, 12, state.upsampledHeight, state.upsampledWidth]),
            arr_last_features_reweighted,
            weights,
            biases
        )
    }
}

export let build = async (state, outputOperand) => {
    let graph = await state.builder.build({ 'output': outputOperand })

    return {
        ...state,
        graph
    }
}

export let compute = async (state,
    [view_tensor1, view_tensor2, view_tensor3, view_tensor4, view_tensor5, view_tensor6,],
    [depth_tensor1, depth_tensor2, depth_tensor3, depth_tensor4, depth_tensor5, depth_tensor6,],
    output) => {
    let inputs = {
        'input_view1': view_tensor1,
        'input_view2': view_tensor2,
        'input_view3': view_tensor3,
        'input_view4': view_tensor4,
        'input_view5': view_tensor5,
        'input_view6': view_tensor6,
        'input_depth1': depth_tensor1,
        'input_depth2': depth_tensor2,
        'input_depth3': depth_tensor3,
        'input_depth4': depth_tensor4,
        'input_depth5': depth_tensor5,
        'input_depth6': depth_tensor6,
    }
    let outputs = { 'output': output }
    let results = await state.context.compute(state.graph, inputs, outputs);

    return results;
}


