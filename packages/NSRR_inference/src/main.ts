import { buildConstantByNpy, sizeOfShape } from "./common/utils"
import { build, compute, createComputeGraphOfInput, createComputeGraphOfFeatureExtract, createState, init, createComputeGraphOfZeroUpsampling, createComputeGraphOfFeatureReweighting, createComputeGraphOfReconstruction } from "./nsrr"
import { loadInputs } from "./input"
import featureExtractSeq_0_0_weight from './checkpoints/featureExtractSeq.0.0.weight.npy'
import featureExtractSeq_0_0_bias from './checkpoints/featureExtractSeq.0.0.bias.npy'
import featureExtractSeq_0_2_weight from './checkpoints/featureExtractSeq.0.2.weight.npy'
import featureExtractSeq_0_2_bias from './checkpoints/featureExtractSeq.0.2.bias.npy'
import featureExtractSeq_0_4_weight from './checkpoints/featureExtractSeq.0.4.weight.npy'
import featureExtractSeq_0_4_bias from './checkpoints/featureExtractSeq.0.4.bias.npy'
import featureExtractSeq_1_0_weight from './checkpoints/featureExtractSeq.1.0.weight.npy'
import featureExtractSeq_1_0_bias from './checkpoints/featureExtractSeq.1.0.bias.npy'
import featureExtractSeq_1_2_weight from './checkpoints/featureExtractSeq.1.2.weight.npy'
import featureExtractSeq_1_2_bias from './checkpoints/featureExtractSeq.1.2.bias.npy'
import featureExtractSeq_1_4_weight from './checkpoints/featureExtractSeq.1.4.weight.npy'
import featureExtractSeq_1_4_bias from './checkpoints/featureExtractSeq.1.4.bias.npy'
import featureExtractSeq_2_0_weight from './checkpoints/featureExtractSeq.2.0.weight.npy'
import featureExtractSeq_2_0_bias from './checkpoints/featureExtractSeq.2.0.bias.npy'
import featureExtractSeq_2_2_weight from './checkpoints/featureExtractSeq.2.2.weight.npy'
import featureExtractSeq_2_2_bias from './checkpoints/featureExtractSeq.2.2.bias.npy'
import featureExtractSeq_2_4_weight from './checkpoints/featureExtractSeq.2.4.weight.npy'
import featureExtractSeq_2_4_bias from './checkpoints/featureExtractSeq.2.4.bias.npy'
import featureExtractSeq_3_0_weight from './checkpoints/featureExtractSeq.3.0.weight.npy'
import featureExtractSeq_3_0_bias from './checkpoints/featureExtractSeq.3.0.bias.npy'
import featureExtractSeq_3_2_weight from './checkpoints/featureExtractSeq.3.2.weight.npy'
import featureExtractSeq_3_2_bias from './checkpoints/featureExtractSeq.3.2.bias.npy'
import featureExtractSeq_3_4_weight from './checkpoints/featureExtractSeq.3.4.weight.npy'
import featureExtractSeq_3_4_bias from './checkpoints/featureExtractSeq.3.4.bias.npy'
import featureExtractSeq_4_0_weight from './checkpoints/featureExtractSeq.4.0.weight.npy'
import featureExtractSeq_4_0_bias from './checkpoints/featureExtractSeq.4.0.bias.npy'
import featureExtractSeq_4_2_weight from './checkpoints/featureExtractSeq.4.2.weight.npy'
import featureExtractSeq_4_2_bias from './checkpoints/featureExtractSeq.4.2.bias.npy'
import featureExtractSeq_4_4_weight from './checkpoints/featureExtractSeq.4.4.weight.npy'
import featureExtractSeq_4_4_bias from './checkpoints/featureExtractSeq.4.4.bias.npy'
import featureExtractSeq_5_0_weight from './checkpoints/featureExtractSeq.5.0.weight.npy'
import featureExtractSeq_5_0_bias from './checkpoints/featureExtractSeq.5.0.bias.npy'
import featureExtractSeq_5_2_weight from './checkpoints/featureExtractSeq.5.2.weight.npy'
import featureExtractSeq_5_2_bias from './checkpoints/featureExtractSeq.5.2.bias.npy'
import featureExtractSeq_5_4_weight from './checkpoints/featureExtractSeq.5.4.weight.npy'
import featureExtractSeq_5_4_bias from './checkpoints/featureExtractSeq.5.4.bias.npy'
import feature_reweighting_model_weighting_0_weight from './checkpoints/feature_reweighting_model.weighting.0.weight.npy'
import feature_reweighting_model_weighting_0_bias from './checkpoints/feature_reweighting_model.weighting.0.bias.npy'
import feature_reweighting_model_weighting_2_weight from './checkpoints/feature_reweighting_model.weighting.2.weight.npy'
import feature_reweighting_model_weighting_2_bias from './checkpoints/feature_reweighting_model.weighting.2.bias.npy'
import feature_reweighting_model_weighting_4_weight from './checkpoints/feature_reweighting_model.weighting.4.weight.npy'
import feature_reweighting_model_weighting_4_bias from './checkpoints/feature_reweighting_model.weighting.4.bias.npy'
import reconstructionModel_encoder_1_0_weight from './checkpoints/reconstructionModel.encoder_1.0.weight.npy'
import reconstructionModel_encoder_1_0_bias from './checkpoints/reconstructionModel.encoder_1.0.bias.npy'
import reconstructionModel_encoder_1_2_weight from './checkpoints/reconstructionModel.encoder_1.2.weight.npy'
import reconstructionModel_encoder_1_2_bias from './checkpoints/reconstructionModel.encoder_1.2.bias.npy'
import reconstructionModel_encoder_2_0_weight from './checkpoints/reconstructionModel.encoder_2.0.weight.npy'
import reconstructionModel_encoder_2_0_bias from './checkpoints/reconstructionModel.encoder_2.0.bias.npy'
import reconstructionModel_encoder_2_2_weight from './checkpoints/reconstructionModel.encoder_2.2.weight.npy'
import reconstructionModel_encoder_2_2_bias from './checkpoints/reconstructionModel.encoder_2.2.bias.npy'
import reconstructionModel_center_0_weight from './checkpoints/reconstructionModel.center.0.weight.npy'
import reconstructionModel_center_0_bias from './checkpoints/reconstructionModel.center.0.bias.npy'
import reconstructionModel_center_2_weight from './checkpoints/reconstructionModel.center.2.weight.npy'
import reconstructionModel_center_2_bias from './checkpoints/reconstructionModel.center.2.bias.npy'
import reconstructionModel_decoder_2_0_weight from './checkpoints/reconstructionModel.decoder_2.0.weight.npy'
import reconstructionModel_decoder_2_0_bias from './checkpoints/reconstructionModel.decoder_2.0.bias.npy'
import reconstructionModel_decoder_2_2_weight from './checkpoints/reconstructionModel.decoder_2.2.weight.npy'
import reconstructionModel_decoder_2_2_bias from './checkpoints/reconstructionModel.decoder_2.2.bias.npy'
import reconstructionModel_decoder_1_0_weight from './checkpoints/reconstructionModel.decoder_1.0.weight.npy'
import reconstructionModel_decoder_1_0_bias from './checkpoints/reconstructionModel.decoder_1.0.bias.npy'
import reconstructionModel_cat_1_0_weight from './checkpoints/reconstructionModel.cat_1.0.weight.npy'
import reconstructionModel_cat_1_0_bias from './checkpoints/reconstructionModel.cat_1.0.bias.npy'
import reconstructionModel_cat_2_0_weight from './checkpoints/reconstructionModel.cat_2.0.weight.npy'
import reconstructionModel_cat_2_0_bias from './checkpoints/reconstructionModel.cat_2.0.bias.npy'



let _drawOutput = (outputBuffer, [width, height]) => {
    const mean = [1, 1, 1, 1];
    const offset = [0, 0, 0, 0];
    const bytes = new Uint8ClampedArray(width * height * 4);
    const a = 255;

    for (let i = 0; i < height * width; ++i) {
        const j = i * 4;
        const r = outputBuffer[i] * mean[0] + offset[0];
        const g = outputBuffer[i + height * width] * mean[1] + offset[1];
        const b = outputBuffer[i + height * width * 2] * mean[2] + offset[2];
        bytes[j + 0] = Math.round(r * 255);
        bytes[j + 1] = Math.round(g * 255);
        bytes[j + 2] = Math.round(b * 255);
        bytes[j + 3] = Math.round(a);
    }

    const imageData = new ImageData(bytes, width, height);
    const outCanvas = document.createElement('canvas');
    const outCtx = outCanvas.getContext('2d');
    outCanvas.width = width;
    outCanvas.height = height;
    outCtx.putImageData(imageData, 0, 0, 0, 0, outCanvas.width, outCanvas.height);

    document.body.append(outCanvas)
}

window.onload = async () => {
    let downscale = [4, 4]
    let [upsampledWidth, upsampledHeight] = [720, 480]
    let [width, height] = [upsampledWidth / downscale[0], upsampledHeight / downscale[1]]

    let state = createState([width, height])

    let [
        view_tensors,
        depth_tensors
    ] = await loadInputs([width, height])
    // ] = await loadInputs([upsampledWidth, upsampledHeight])

    state = await init(state, {
        deviceType: "gpu"
    })

    // console.log(view_tensors)
    // let view_tensor = state.builder.concat(view_tensors, 0)
    // let depth_tensor = state.builder.concat(depth_tensors, 0)


    state = createComputeGraphOfInput(state)
    state = createComputeGraphOfFeatureExtract(state,
        [
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_0_0_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_0_2_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_0_4_weight),
            ],
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_1_0_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_1_2_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_1_4_weight),
            ],
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_2_0_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_2_2_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_2_4_weight),
            ],
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_3_0_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_3_2_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_3_4_weight),
            ],
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_4_0_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_4_2_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_4_4_weight),
            ],
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_5_0_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_5_2_weight),
                await buildConstantByNpy(state.builder, featureExtractSeq_5_4_weight),
            ],
        ],
        [
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_0_0_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_0_2_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_0_4_bias),
            ],
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_1_0_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_1_2_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_1_4_bias),
            ],
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_2_0_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_2_2_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_2_4_bias),
            ],
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_3_0_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_3_2_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_3_4_bias),
            ],
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_4_0_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_4_2_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_4_4_bias),
            ],
            [
                await buildConstantByNpy(state.builder, featureExtractSeq_5_0_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_5_2_bias),
                await buildConstantByNpy(state.builder, featureExtractSeq_5_4_bias),
            ]
        ]
    )
    state = createComputeGraphOfZeroUpsampling(state)
    state = createComputeGraphOfFeatureReweighting(state, [
        await buildConstantByNpy(state.builder, feature_reweighting_model_weighting_0_weight),
        await buildConstantByNpy(state.builder, feature_reweighting_model_weighting_2_weight),
        await buildConstantByNpy(state.builder, feature_reweighting_model_weighting_4_weight),
    ],
        [

            await buildConstantByNpy(state.builder, feature_reweighting_model_weighting_0_bias),
            await buildConstantByNpy(state.builder, feature_reweighting_model_weighting_2_bias),
            await buildConstantByNpy(state.builder, feature_reweighting_model_weighting_4_bias),
        ])
    state = createComputeGraphOfReconstruction(state,
        [
            [
                await buildConstantByNpy(state.builder, reconstructionModel_encoder_1_0_weight),
                await buildConstantByNpy(state.builder, reconstructionModel_encoder_1_2_weight),
            ],
            [
                await buildConstantByNpy(state.builder, reconstructionModel_encoder_2_0_weight),
                await buildConstantByNpy(state.builder, reconstructionModel_encoder_2_2_weight),
            ],
            [
                await buildConstantByNpy(state.builder, reconstructionModel_center_0_weight),
                await buildConstantByNpy(state.builder, reconstructionModel_center_2_weight),
            ],

            await buildConstantByNpy(state.builder, reconstructionModel_cat_1_0_weight),

            [

                await buildConstantByNpy(state.builder, reconstructionModel_decoder_2_0_weight),
                await buildConstantByNpy(state.builder, reconstructionModel_decoder_2_2_weight),
            ],

            await buildConstantByNpy(state.builder, reconstructionModel_cat_2_0_weight),


            await buildConstantByNpy(state.builder, reconstructionModel_decoder_1_0_weight)

        ],
        [
            [
                await buildConstantByNpy(state.builder, reconstructionModel_encoder_1_0_bias),
                await buildConstantByNpy(state.builder, reconstructionModel_encoder_1_2_bias),
            ],
            [
                await buildConstantByNpy(state.builder, reconstructionModel_encoder_2_0_bias),
                await buildConstantByNpy(state.builder, reconstructionModel_encoder_2_2_bias),
            ],
            [
                await buildConstantByNpy(state.builder, reconstructionModel_center_0_bias),
                await buildConstantByNpy(state.builder, reconstructionModel_center_2_bias),
            ],

            await buildConstantByNpy(state.builder, reconstructionModel_cat_1_0_bias),

            [

                await buildConstantByNpy(state.builder, reconstructionModel_decoder_2_0_bias),
                await buildConstantByNpy(state.builder, reconstructionModel_decoder_2_2_bias),
            ],

            await buildConstantByNpy(state.builder, reconstructionModel_cat_2_0_bias),

            await buildConstantByNpy(state.builder, reconstructionModel_decoder_1_0_bias)

        ])

    state = await build(state, state.output)

    let outputBuffer = new Float32Array(sizeOfShape([1, 3, upsampledWidth, upsampledHeight]));

    let results
    //TODO warm up
    // results = await compute(state, view_tensors, depth_tensors, outputBuffer)

    let start = performance.now();
    results = await compute(state, view_tensors, depth_tensors, outputBuffer)
    console.log(performance.now() - start)


    console.log(results.outputs.output)

    _drawOutput(results.outputs.output, [upsampledWidth, upsampledHeight])
}