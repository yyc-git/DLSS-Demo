import { sizeOfShape } from "../../src/common/utils";

export let computeWithNoInput = async (outputDimensions, context, graph) => {
    let outputBuffer = new Float32Array(sizeOfShape(outputDimensions));

    let inputs = {}
    let outputs = { 'output': outputBuffer }

    return await context.compute(graph, inputs, outputs);
}