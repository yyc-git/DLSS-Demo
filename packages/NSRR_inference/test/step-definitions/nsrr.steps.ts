import { loadFeature, defineFeature } from 'jest-cucumber';
import { sizeOfShape } from '../../src/common/utils';
import { build, compute, createComputeGraphOfFeatureExtract, createComputeGraphOfFeatureReweighting, createComputeGraphOfInput, createComputeGraphOfReconstruction, createComputeGraphOfZeroUpsampling, createState, _prepareWeightForZeroUpsampling, _remap, _zeroUpsampling } from '../../src/nsrr';
import { computeWithNoInput } from '../utils/ComputeUtils';
import { buildTensor, convertImageDataToFloat32Tensor } from '../utils/TensorUtils';

const feature = loadFeature('./test/features/nsrr.feature');

defineFeature(feature, test => {
    let state
    let context, builder
    let results

    // TODO move to utils
    function _buildWeight(dimensions, value) {
        return builder.constant({ type: "float32", dimensions: dimensions },
            new Float32Array(sizeOfShape(dimensions)).fill(value)
        )
    }

    function _buildBias(outputChannels, value) {
        let dimensions = [1, 1, 1, outputChannels]

        return builder.constant({ type: "float32", dimensions: dimensions },
            new Float32Array(sizeOfShape(dimensions)).fill(value)
        )
    }

    function _prepare(given, and) {
        given('create context', async () => {
            context = await (navigator as any).ml.createContext({
                deviceType: "cpu"
            })
        })

        and('set backend to cpu', async () => {
            let tf = context.tf
            await tf.setBackend("cpu")
            await tf.ready()
        })


        and('create builder', () => {
            builder = new MLGraphBuilder(context)
        })

    }

    test('zero upsampling', ({
        given,
        and,
        when,
        then
    }) => {
        let width = 1
        let height = 2
        let channel = 2
        let n = 2

        _prepare(given, and)

        given('create state with fake all_features', () => {
            state = createState([width, height])
            state = {
                ...state,
                context,
                builder,
                all_features: buildTensor(
                    builder,
                    new Float32Array([
                        1, 2,
                        3, 4,

                        2, 3,
                        4, 5,
                    ]),
                    [n, channel, height, width]
                )
            }
        })

        when('zero upsampling', () => {
            state = {
                ...state,
                all_features_upsampled: _zeroUpsampling(builder, state.all_features,
                    _prepareWeightForZeroUpsampling(state, channel),
                    n, channel, height, width)
            }
        })

        and('build', async () => {
            state = await build(state, state.all_features_upsampled)
        })

        and('compute with no input', async () => {
            results = await computeWithNoInput([n, channel, height * 4, width * 4], state.context, state.graph)
        });

        then('get zero upsampling data', () => {
            expect(results.outputs.output).toEqual(new Float32Array([
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                2,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                3,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                4,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                5,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ]))
        });
    });

    test('remap', ({
        given,
        and,
        when,
        then
    }) => {
        let width = 1
        let height = 2
        let channel = 2
        let n = 1
        let tensor

        _prepare(given, and)

        given('prepare tensor in range [-1,1]', () => {
            tensor = buildTensor(
                builder,
                new Float32Array([
                    0.0, -1.0,
                    1.0, 0.5,
                ]),
                [n, channel, height, width]
            )
        })

        and('create state', () => {
            state = createState([width, height])
            state = {
                ...state,
                context,
                builder
            }
        })

        when('remap tensor to range [0,10]', () => {
            tensor = _remap(builder, tensor, [n, channel, height, width], [-1, 1], [0, 10])
        })

        and('build', async () => {
            state = await build(state, tensor)
        })

        and('compute with no input', async () => {
            results = await computeWithNoInput([n, channel, height, width], state.context, state.graph)
        });

        then('get remaped data in range [0,10]', () => {
            expect(results.outputs.output).toEqual(new Float32Array([
                5,
                0,
                10,
                7.5,
            ]))
        });
    });

    test('create compute graph of whole', ({
        given,
        and,
        when,
        then
    }) => {
        let width = 1
        let height = 2
        let kernelSize = 3
        let view_tensor1, view_tensor2, view_tensor3, view_tensor4, view_tensor5, view_tensor6, depth_tensor1, depth_tensor2, depth_tensor3, depth_tensor4, depth_tensor5, depth_tensor6

        _prepare(given, and)

        given('prepare fake input: view_tensor, depth_tensor', () => {
            view_tensor1 = convertImageDataToFloat32Tensor(
                [
                    0.5, 1.0, 0.5, 0.1,
                    1.0, 0.5, 0.5, 0.3,
                ],
                [1, 3, height, width]
            )
            view_tensor2 = convertImageDataToFloat32Tensor(
                [
                    0.5, 1.0, 0.5, 0.1,
                    1.0, 0.5, 0.5, 0.3,
                ],
                [1, 3, height, width]
            )
            view_tensor3 = convertImageDataToFloat32Tensor(
                [
                    0.5, 1.0, 0.5, 0.1,
                    1.0, 0.5, 0.5, 0.3,
                ],
                [1, 3, height, width]
            )
            view_tensor4 = convertImageDataToFloat32Tensor(
                [
                    0.5, 1.0, 0.5, 0.1,
                    1.0, 0.5, 0.5, 0.3,
                ],
                [1, 3, height, width]
            )
            view_tensor5 = convertImageDataToFloat32Tensor(
                [
                    0.5, 1.0, 0.5, 0.1,
                    1.0, 0.5, 0.5, 0.3,
                ],
                [1, 3, height, width]
            )
            view_tensor6 = convertImageDataToFloat32Tensor(
                [
                    0.5, 1.0, 0.5, 0.1,
                    1.0, 0.5, 0.5, 0.3,
                ],
                [1, 3, height, width]
            )
            depth_tensor1 = convertImageDataToFloat32Tensor(
                [
                    1.0, 0.0, 0.0, 1.0,
                    0.5, 0.0, 0.0, 1.0,
                ],
                [1, 1, height, width]
            )
            depth_tensor2 = convertImageDataToFloat32Tensor(
                [
                    1.0, 0.0, 0.0, 1.0,
                    0.5, 0.0, 0.0, 1.0,
                ],
                [1, 1, height, width]
            )
            depth_tensor3 = convertImageDataToFloat32Tensor(
                [
                    1.0, 0.0, 0.0, 1.0,
                    0.5, 0.0, 0.0, 1.0,
                ],
                [1, 1, height, width]
            )
            depth_tensor4 = convertImageDataToFloat32Tensor(
                [
                    1.0, 0.0, 0.0, 1.0,
                    0.5, 0.0, 0.0, 1.0,
                ],
                [1, 1, height, width]
            )
            depth_tensor5 = convertImageDataToFloat32Tensor(
                [
                    1.0, 0.0, 0.0, 1.0,
                    0.5, 0.0, 0.0, 1.0,
                ],
                [1, 1, height, width]
            )
            depth_tensor6 = convertImageDataToFloat32Tensor(
                [
                    1.0, 0.0, 0.0, 1.0,
                    0.5, 0.0, 0.0, 1.0,
                ],
                [1, 1, height, width]
            )
        })

        and('create state', () => {
            state = createState([width, height])
            state = {
                ...state,
                context,
                builder,
            }
        })

        and('prepare weightForZeroUpsampling', () => {
            state = {
                ...state,
                weightForZeroUpsamplingAllFeatures: _prepareWeightForZeroUpsampling(state, 12),
                weightForZeroUpsamplingCurrentFrame: _prepareWeightForZeroUpsampling(state, 4),
            }
        })

        when('create compute graph of input', () => {
            state = createComputeGraphOfInput(state)
        })

        and('create compute graph of feature extract', () => {
            state = createComputeGraphOfFeatureExtract(state,
                [
                    [
                        _buildWeight(
                            [32, 4, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [32, 32, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [8, 32, kernelSize, kernelSize],
                            0.5
                        ),
                    ],
                    [
                        _buildWeight(
                            [32, 4, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [32, 32, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [8, 32, kernelSize, kernelSize],
                            0.5
                        ),
                    ],
                    [
                        _buildWeight(
                            [32, 4, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [32, 32, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [8, 32, kernelSize, kernelSize],
                            0.5
                        ),
                    ],
                    [
                        _buildWeight(
                            [32, 4, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [32, 32, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [8, 32, kernelSize, kernelSize],
                            0.5
                        ),
                    ],
                    [
                        _buildWeight(
                            [32, 4, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [32, 32, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [8, 32, kernelSize, kernelSize],
                            0.5
                        ),
                    ],
                    [
                        _buildWeight(
                            [32, 4, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [32, 32, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [8, 32, kernelSize, kernelSize],
                            0.5
                        ),
                    ],
                ],
                [
                    [
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            8,
                            1.0
                        ),
                    ],
                    [
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            8,
                            1.0
                        ),
                    ],

                    [
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            8,
                            1.0
                        ),
                    ],

                    [
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            8,
                            1.0
                        ),
                    ],

                    [
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            8,
                            1.0
                        ),
                    ],

                    [
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            32,
                            0.5
                        ),
                        _buildBias(
                            8,
                            1.0
                        ),
                    ],
                ]
            )
        })

        and('create compute graph of zero upsampling', () => {
            state = createComputeGraphOfZeroUpsampling(state)
        })

        and('create compute graph of feature reweighting', () => {
            state = createComputeGraphOfFeatureReweighting(state,
                [
                    _buildWeight(
                        [40, 24, kernelSize, kernelSize],
                        1.0
                    ),
                    _buildWeight(
                        [40, 40, kernelSize, kernelSize],
                        1.0
                    ),
                    _buildWeight(
                        [5, 40, kernelSize, kernelSize],
                        0.5
                    ),
                ],
                [
                    _buildBias(40, 1.0),
                    _buildBias(40, 1.0),
                    _buildBias(5, 1.0),
                ]
            )
        })

        and('create compute graph of reconstruction', () => {
            state = createComputeGraphOfReconstruction(state,
                [
                    [

                        _buildWeight(
                            [64, 72, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [32, 64, kernelSize, kernelSize],
                            1.0
                        ),
                    ],
                    [
                        _buildWeight(
                            [64, 32, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [64, 64, kernelSize, kernelSize],
                            2.0
                        ),
                    ],
                    [
                        _buildWeight(
                            [128, 64, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [128, 128, kernelSize, kernelSize],
                            2.0
                        ),
                    ],
                    _buildWeight(
                        [128, 192, kernelSize, kernelSize],
                        1.0
                    ),
                    [
                        _buildWeight(
                            [64, 128, kernelSize, kernelSize],
                            1.0
                        ),
                        _buildWeight(
                            [64, 64, kernelSize, kernelSize],
                            1.0
                        ),
                    ],
                    _buildWeight(
                        [32, 96, kernelSize, kernelSize],
                        1.0
                    ),
                    _buildWeight(
                        [3, 32, kernelSize, kernelSize],
                        1.0
                    ),
                ],
                [
                    [
                        _buildBias(
                            64,
                            1.0
                        ),
                        _buildBias(
                            32,
                            1.0
                        ),
                    ],
                    [
                        _buildBias(
                            64,
                            1.0
                        ),
                        _buildBias(
                            64,
                            2.0
                        ),
                    ],
                    [
                        _buildBias(
                            128,
                            1.0
                        ),
                        _buildBias(
                            128,
                            2.0
                        ),
                    ],
                    _buildBias(
                        128,
                        1.0
                    ),
                    [
                        _buildBias(
                            64,
                            1.0
                        ),
                        _buildBias(
                            64,
                            1.0
                        ),
                    ],
                    _buildBias(
                        32,
                        1.0
                    ),
                    _buildBias(
                        3,
                        1.0
                    ),
                ]
            )
        })

        and('build', async () => {
            state = await build(state, state.output)
        })

        and('compute with input', async () => {
            let outputBuffer = new Float32Array(sizeOfShape([1, 3, height * 4, width * 4]));

            results = await compute(state,
                [view_tensor1, view_tensor2, view_tensor3, view_tensor4, view_tensor5, view_tensor6,],
                [depth_tensor1, depth_tensor2, depth_tensor3, depth_tensor4, depth_tensor5, depth_tensor6,],
                outputBuffer)
        });

        then('get correct data', () => {
            expect(results.outputs.output).toEqual(new Float32Array([
                +   5.667957310471226e+31,
                +   9.068735758744716e+31,
                +   9.06872802161947e+31,
                +   5.66795199119762e+31,
                +   9.824454631398934e+31,
                +   1.5719135340791671e+32,
                +   1.5719127603666426e+32,
                +   9.824474941352704e+31,
                +   1.2318360277510477e+32,
                +   1.970939810796745e+32,
                +   1.9709401976530073e+32,
                +   1.2318365113213755e+32,
                +   1.3376387009180152e+32,
                +   2.140214571199841e+32,
                +   2.140217472621808e+32,
                +   1.3376390877742775e+32,
                +   1.337637830491425e+32,
                +   2.140221534612562e+32,
                +   2.1402205674719064e+32,
                +   1.3376368633507694e+32,
                +   1.231837962032359e+32,
                +   1.970940004224876e+32,
                +   1.9709407779374007e+32,
                +   1.2318383488886213e+32,
                +   9.824466237086802e+31,
                +   1.5719179829261833e+32,
                +   1.5719184664965111e+32,
                +   9.824460434242868e+31,
                +   5.667955376189915e+31,
                +   9.068743495869961e+31,
                +   9.068732857322748e+31,
                +   5.66795876118221e+31,
                +   5.667957310471226e+31,
                +   9.068735758744716e+31,
                +   9.06872802161947e+31,
                +   5.66795199119762e+31,
                +   9.824454631398934e+31,
                +   1.5719135340791671e+32,
                +   1.5719127603666426e+32,
                +   9.824474941352704e+31,
                +   1.2318360277510477e+32,
                +   1.970939810796745e+32,
                +   1.9709401976530073e+32,
                +   1.2318365113213755e+32,
                +   1.3376387009180152e+32,
                +   2.140214571199841e+32,
                +   2.140217472621808e+32,
                +   1.3376390877742775e+32,
                +   1.337637830491425e+32,
                +   2.140221534612562e+32,
                +   2.1402205674719064e+32,
                +   1.3376368633507694e+32,
                +   1.231837962032359e+32,
                +   1.970940004224876e+32,
                +   1.9709407779374007e+32,
                +   1.2318383488886213e+32,
                +   9.824466237086802e+31,
                +   1.5719179829261833e+32,
                +   1.5719184664965111e+32,
                +   9.824460434242868e+31,
                +   5.667955376189915e+31,
                +   9.068743495869961e+31,
                +   9.068732857322748e+31,
                +   5.66795876118221e+31,
                +   5.667957310471226e+31,
                +   9.068735758744716e+31,
                +   9.06872802161947e+31,
                +   5.66795199119762e+31,
                +   9.824454631398934e+31,
                +   1.5719135340791671e+32,
                +   1.5719127603666426e+32,
                +   9.824474941352704e+31,
                +   1.2318360277510477e+32,
                +   1.970939810796745e+32,
                +   1.9709401976530073e+32,
                +   1.2318365113213755e+32,
                +   1.3376387009180152e+32,
                +   2.140214571199841e+32,
                +   2.140217472621808e+32,
                +   1.3376390877742775e+32,
                +   1.337637830491425e+32,
                +   2.140221534612562e+32,
                +   2.1402205674719064e+32,
                +   1.3376368633507694e+32,
                +   1.231837962032359e+32,
                +   1.970940004224876e+32,
                +   1.9709407779374007e+32,
                +   1.2318383488886213e+32,
                +   9.824466237086802e+31,
                +   1.5719179829261833e+32,
                +   1.5719184664965111e+32,
                +   9.824460434242868e+31,
                +   5.667955376189915e+31,
                +   9.068743495869961e+31,
                +   9.068732857322748e+31,
                +   5.66795876118221e+31,
            ]))
        });
    });
});
