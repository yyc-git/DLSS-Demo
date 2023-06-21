import '@webmachinelearning/webnn-polyfill'
import { range } from './common/Array'
import { buildConstantByNpy, sizeOfShape } from "./common/utils"
import { height, Filter, Tensor, width, state, upsampledHeight, upsampledWidth, MLOprand, pool1Width, pool1Height, pool2Height, pool2Width } from './type'

//create state

export let createState = (): state => {
    return {
        context: null,
        builder: null,
        graph: null,
        input_view: null,
        input_depth: null,
        all_features: null,
        all_features_upsampled: null,
        arr_last_features_reweighted: null,
        output: null
    }
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

    return {
        ...state,
        context,
        builder,
    }
}

export let createComputeGraphOfInput = (state, [width, height]) => {
    let { builder } = state

    let input_viewShape = [6, 3, height, width]
    let input_view = builder.input('input_view', { type: 'float32', dimensions: input_viewShape })

    let input_depthShape = [6, 1, height, width]
    let input_depth = builder.input('input_depth', { type: 'float32', dimensions: input_depthShape })

    return {
        ...state,
        input_view,
        input_depth,
    }
}

let _getPadding = () => [1, 1, 1, 1]

let _getStrides = () => [1, 1]

let _buildFeatureExtractModel = (builder, input_rgbd: Tensor<1, 4, height, width>, [conv1Weight, conv2Weight, conv3Weight]: [
    Filter<32, 4, height, width>,
    Filter<32, 32, height, width>,
    Filter<32, 8, height, width>,
]): Tensor<1, 12, height, width> => {
    let conv1 = builder.conv2d(
        input_rgbd,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )
    let conv2 = builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )
    let conv3 = builder.conv2d(
        conv2,
        conv3Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )

    return builder.concat([conv3, input_rgbd], 1)
}

export let createComputeGraphOfFeatureExtract = (state, weights): Tensor<6, 12, height, width> => {
    let {
        builder,
        input_view,
        input_depth
    } = state

    let input_all_rgbd: Tensor<6, 4, height, width> = builder.concat([input_view, input_depth], 1)

    let all_features = builder.concat(
        range(0, 6 - 1).map(i => {
            _buildFeatureExtractModel(builder, builder.slice(input_all_rgbd, [i], [1], { axes: [0] }), weights)
        }), 0
    )

    return {
        ...state,
        all_features
    }
}

//TODO need bdd test
export let createComputeGraphOfZeroUpsampling = (state: state) => {
    let {
        builder,
        all_features
    } = state

    // refer to: https://github.com/pytorch/pytorch/issues/7911#issuecomment-392835113

    let dimension = [1, 1, 4, 4]
    let value = new Float32Array(sizeOfShape(dimension)).fill(0.0)
    value[0] = 1.0
    let weightArr = range(0, 11).reduce((result, i) => {
        result.push(
            builder.constant(
                { type: 'float32', dimensions: dimension },
                value
            )
        )

        return result
    }, [])
    let weight: Filter<1, 12, 4, 4> = builder.concat(weightArr, 1)

    let all_features_upsampled: Tensor<6, 12, upsampledHeight, upsampledWidth> = builder.concat(
        range(0, 6 - 1).map(i => {
            return builder.convTranspose2d(
                builder.slice(all_features, [i], [1], { axes: 0 }),
                weight,
                {
                    strides: [4, 4],
                    groups: 12
                }
            )
        }),
        0
    )

    return {
        ...state,
        all_features_upsampled: all_features_upsampled
    }
}

// TODO need bdd test
let _remap = (builder, x: MLOprand, in_range: [number, number], out_range: [number, number]) => {
    return builder.div(
        builder.mul(builder.add(x, - in_range[0]), out_range[1] - out_range[0]),
        (in_range[1] - in_range[0]) + out_range[0]
    )
}

let _multiplyWeightMap = (builder, weighting_map: Tensor<1, 5, upsampledHeight, upsampledWidth>, input_last_frame_features_upsampled: Tensor<5, 12, upsampledHeight, upsampledWidth>): Array<Tensor<1, 12, upsampledHeight, upsampledWidth>> => {
    return range(0, 6 - 1).reduce((arr, i) => {
        let input_last_frame_feature_upsampled: Tensor<1, 12, upsampledHeight, upsampledWidth> = builder.slice(input_last_frame_features_upsampled, [i], [1], { axes: [0] })

        let weighting_one_last_frame_map: Tensor<1, 1, upsampledHeight, upsampledWidth> = builder.slice(weighting_map, [i], [1], { axes: [1] })

        let tmp: Array<Tensor<1, 1, upsampledHeight, upsampledWidth>> = range(0, 12 - 1).reduce((tmp, j) => {
            tmp.push(
                builder.mul(
                    builder.slice(input_last_frame_feature_upsampled, [j], [1], { axes: [1] }),
                    weighting_one_last_frame_map
                )
            )

            return tmp
        }, [])

        arr.push(builder.concat(tmp, 1))

        return arr
    }, [])
}

let _buildFeatureReweightingModel = (builder,
    input_current_frame_upsampled: Tensor<1, 4, upsampledHeight, upsampledWidth>,
    input_last_frame_features_upsampled: Tensor<5, 12, upsampledHeight, upsampledWidth>,
    [conv1Weight, conv2Weight, conv3Weight]: [
        Filter<40, 24, upsampledHeight, upsampledWidth>,
        Filter<40, 40, upsampledHeight, upsampledWidth>,
        Filter<40, 5, upsampledHeight, upsampledWidth>,
    ]
): Array<Tensor<1, 12, upsampledHeight, upsampledWidth>> => {
    let reweight_feed_in: Tensor<1, 24, upsampledHeight, upsampledWidth> = range(0, 6 - 1).reduce((reweight_feed_in, i) => {
        let input_last_frame_feature_upsampled = builder.slice(input_last_frame_features_upsampled, [i], [1], { axes: [0] })

        return builder.concat(
            [
                reweight_feed_in,
                builder.slice(input_last_frame_feature_upsampled, [0], [4], { axes: [1] })
            ], 0
        )
    }, input_current_frame_upsampled)

    let conv1 = builder.conv2d(
        reweight_feed_in,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )
    let conv2 = builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )
    let conv3 = builder.conv2d(
        conv2,
        conv3Weight,
        {
            activation: builder.tanh(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )

    let weighting_map: Tensor<1, 5, upsampledHeight, upsampledWidth> = _remap(builder, conv3, [-1, 1], [0, 10])

    return _multiplyWeightMap(builder, weighting_map, input_last_frame_features_upsampled)
}

export let createComputeGraphOfFeatureReweighting = (state, weights) => {
    let {
        builder,
        input_view,
        all_features_upsampled
    } = state

    let input_current_frame_upsampled =
        builder.resample2d(builder.slice(input_view, [0], [1], { axes: 0 }), {
            mode: "nearest-neighbor",
            scales: [4, 4],
            axes: [2, 3]
        })
    let input_last_frame_features_upsampled = builder.slice(all_features_upsampled, [1], [5], { axes: 0 })

    let arr_last_features_reweighted = _buildFeatureReweightingModel(builder,
        input_current_frame_upsampled,
        input_last_frame_features_upsampled,
        weights
    )

    return {
        ...state,
        arr_last_features_reweighted: arr_last_features_reweighted
    }
}

let _builderEncoder1 = (builder, input: Tensor<1, 72, upsampledHeight, upsampledWidth>, [conv1Weight, conv2Weight]: [Filter<64, 72, upsampledHeight, upsampledWidth>,
    Filter<32, 64, upsampledHeight, upsampledWidth>
]): Tensor<1, 32, upsampledHeight, upsampledWidth> => {
    let conv1 = builder.conv2d(
        input,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )

    return builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )
}

let _builderEncoder2 = (builder, input: Tensor<1, 32, pool1Height, pool1Width>, [conv1Weight, conv2Weight]: [Filter<64, 32, pool1Height, pool1Width>,
    Filter<64, 64, pool1Height, pool1Width>
]): Tensor<1, 64, pool1Height, pool1Width> => {
    let conv1 = builder.conv2d(
        input,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )

    return builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )
}

let _builderCenter = (builder, input: Tensor<1, 64, pool2Height, pool2Width>, [conv1Weight, conv2Weight]: [Filter<128, 64, pool2Height, pool2Width>,
    Filter<128, 128, pool2Height, pool2Width>
]): Tensor<1, 128, pool1Height, pool1Width> => {
    let conv1 = builder.conv2d(
        input,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )

    let conv2 = builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )

    // refer to https://github.com/webmachinelearning/webnn/issues/358
    return builder.resample2d(conv2, {
        mode: "linear",
        scales: [2, 2],
        axes: [2, 3]
    })
}

let _builderCat1 = (builder, input: Tensor<1, 192, pool1Height, pool1Width>, convWeight: Filter<128, 192, pool1Height, pool1Width>): Tensor<1, 128, pool1Height, pool1Width> => {
    return builder.conv2d(
        input,
        convWeight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )
}

let _builderDecoder2 = (builder, input: Tensor<1, 128, pool1Height, pool1Width>, [conv1Weight, conv2Weight]: [Filter<64, 128, pool1Height, pool1Width>,
    Filter<64, 64, pool1Height, pool1Width>
]): Tensor<1, 64, upsampledHeight, upsampledWidth> => {
    let conv1 = builder.conv2d(
        input,
        conv1Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )

    let conv2 = builder.conv2d(
        conv1,
        conv2Weight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )

    return builder.resample2d(conv2, {
        mode: "linear",
        scales: [2, 2],
        axes: [2, 3]
    })
}

let _builderCat2 = (builder, input: Tensor<1, 96, upsampledHeight, upsampledWidth>, convWeight: Filter<32, 96, upsampledHeight, upsampledWidth>): Tensor<1, 32, upsampledHeight, upsampledWidth> => {
    return builder.conv2d(
        input,
        convWeight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
        }
    )
}

let _builderDecoder1 = (builder, input: Tensor<1, 32, upsampledHeight, upsampledWidth>, convWeight: Filter<3, 32, upsampledHeight, upsampledWidth>): Tensor<1, 3, upsampledHeight, upsampledWidth> => {
    return builder.conv2d(
        input,
        convWeight,
        {
            activation: builder.relu(),
            padding: _getPadding(),
            strides: _getStrides()
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

    let x_encoder_1 = _builderEncoder1(builder, x, encoder1Weight)
    x = builder.maxPool2d(x_encoder_1, {
        windowDimensions: [2, 2],
        strides: [2, 2]
    })
    let x_encoder_2 = _builderEncoder2(builder, x, encoder2Weight)
    x = builder.maxPool2d(x_encoder_2, {
        windowDimensions: [2, 2],
        strides: [2, 2]
    })
    x = _builderCenter(builder, x, centerWeight)

    x = builder.concat([x, x_encoder_2], 1)
    x = _builderCat1(builder, x, cat1Weight)
    x = _builderDecoder2(builder, x, decoder2Weight)


    x = builder.concat([x, x_encoder_1], 1)
    x = _builderCat2(builder, x, cat2Weight)
    x = _builderDecoder1(builder, x, decoder1Weight)

    return x
}

export let createComputeGraphOfReconstruction = (state: state, weights) => {
    let {
        builder,
        all_features_upsampled,
        arr_last_features_reweighted
    } = state

    return {
        ...state,
        output: _buildFeatureReconstructionModel(builder,
            builder.slice(all_features_upsampled, [0], [1], { axes: 0 }),
            arr_last_features_reweighted,
            weights
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

export let compute = async (state, view_tensor, depth_tensor, output) => {
    let inputs = {
        'input_view': view_tensor,
        'input_depth': depth_tensor,
    }
    let outputs = { 'output': output }
    let results = await state.context.compute(state.graph, inputs, outputs);

    return results;
}

