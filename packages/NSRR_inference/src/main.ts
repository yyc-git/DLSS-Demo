import { buildConstantByNpy, sizeOfShape } from "./common/utils"
import { build, compute, createComputeGraphOfInput, createComputeGraphOfFeatureExtract, createState, init, createComputeGraphOfZeroUpsampling, createComputeGraphOfFeatureReweighting, createComputeGraphOfReconstruction } from "./nsrr"
import { loadInputs } from "./input"
// import conv1Weight_path from './checkpoints/open-source-test/classroom/conv1Weight.npy'
// import conv2Weight_path from './checkpoints/open-source-test/classroom/conv2Weight.npy'
// import conv3Weight_path from './checkpoints/open-source-test/classroom/conv3Weight.npy'
// import conv4Weight_path from './checkpoints/open-source-test/classroom/conv4Weight.npy'
// import conv5Weight_path from './checkpoints/open-source-test/classroom/conv5Weight.npy'
// import convFinalWeight_path from './checkpoints/open-source-test/classroom/convFinalWeight.npy'


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
    let [width, height] = [720, 480]
    let [downscaledWidth, downscaledHeight] = [width / downscale[0], height / downscale[1]]

    let state = createState()

    let [
        view_tensors,
        depth_tensors
    ] = await loadInputs([downscaledWidth, downscaledHeight])

    state = await init(state, {
        deviceType: "gpu"
    })

    let view_tensor = state.builder.concat(view_tensors, 0)
    let depth_tensor = state.builder.concat(depth_tensors, 0)


    state = createComputeGraphOfInput(state,
        [downscaledWidth, downscaledHeight]
    )
    state = createComputeGraphOfFeatureExtract(state,
        // [
        //     await buildConstantByNpy(state.builder, conv1Weight_path),
        //     await buildConstantByNpy(state.builder, conv2Weight_path), await buildConstantByNpy(state.builder, conv3Weight_path), await buildConstantByNpy(state.builder, conv4Weight_path), await buildConstantByNpy(state.builder, conv5Weight_path), await buildConstantByNpy(state.builder, convFinalWeight_path),
        // ]
        TODO weights
    )
    state = createComputeGraphOfZeroUpsampling(state,)
    state = createComputeGraphOfFeatureReweighting(state, TODO weights)
    state = createComputeGraphOfReconstruction(state, TODO weights)

    state = await build(state, state.output)

    let outputBuffer = new Float32Array(sizeOfShape([1, 3, width, height]));

    let results
    //warm up
    results = await compute(state, view_tensor, depth_tensor, outputBuffer)

    let start = performance.now();
    results = await compute(state, view_tensor, depth_tensor, outputBuffer)
    console.log(performance.now() - start)


    // console.log(results.outputs.output)

    _drawOutput(results.outputs.output, [width, height])
}