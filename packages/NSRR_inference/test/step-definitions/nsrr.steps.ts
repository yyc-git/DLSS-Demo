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
                5.188491043997802e+31,
                8.301581414977598e+31,
                8.301581414977598e+31,
                5.188495396130753e+31,
                8.993382928837808e+31,
                1.4389430094672295e+32,
                1.4389430094672295e+32,
                8.993380994556496e+31,
                1.1276318446598074e+32,
                1.8042123054526097e+32,
                1.8042134660213966e+32,
                1.127631941373873e+32,
                1.2244786986410042e+32,
                1.959168161591928e+32,
                1.9591691287325836e+32,
                1.2244793756394631e+32,
                1.2244815033489057e+32,
                1.959169515588846e+32,
                1.959172610438944e+32,
                1.2244810197785778e+32,
                1.1276298136644304e+32,
                1.8042074697493313e+32,
                1.8042080500337247e+32,
                1.127629910378496e+32,
                8.993372290290595e+31,
                1.438940881757787e+32,
                1.4389399146171313e+32,
                8.993401304510266e+31,
                5.188482823302229e+31,
                8.301574644993008e+31,
                8.301579480696287e+31,
                5.188481372591245e+31,
                5.188491043997802e+31,
                8.301581414977598e+31,
                8.301581414977598e+31,
                5.188495396130753e+31,
                8.993382928837808e+31,
                1.4389430094672295e+32,
                1.4389430094672295e+32,
                8.993380994556496e+31,
                1.1276318446598074e+32,
                1.8042123054526097e+32,
                1.8042134660213966e+32,
                1.127631941373873e+32,
                1.2244786986410042e+32,
                1.959168161591928e+32,
                1.9591691287325836e+32,
                1.2244793756394631e+32,
                1.2244815033489057e+32,
                1.959169515588846e+32,
                1.959172610438944e+32,
                1.2244810197785778e+32,
                1.1276298136644304e+32,
                1.8042074697493313e+32,
                1.8042080500337247e+32,
                1.127629910378496e+32,
                8.993372290290595e+31,
                1.438940881757787e+32,
                1.4389399146171313e+32,
                8.993401304510266e+31,
                5.188482823302229e+31,
                8.301574644993008e+31,
                8.301579480696287e+31,
                5.188481372591245e+31,
                5.188491043997802e+31,
                8.301581414977598e+31,
                8.301581414977598e+31,
                5.188495396130753e+31,
                8.993382928837808e+31,
                1.4389430094672295e+32,
                1.4389430094672295e+32,
                8.993380994556496e+31,
                1.1276318446598074e+32,
                1.8042123054526097e+32,
                1.8042134660213966e+32,
                1.127631941373873e+32,
                1.2244786986410042e+32,
                1.959168161591928e+32,
                1.9591691287325836e+32,
                1.2244793756394631e+32,
                1.2244815033489057e+32,
                1.959169515588846e+32,
                1.959172610438944e+32,
                1.2244810197785778e+32,
                1.1276298136644304e+32,
                1.8042074697493313e+32,
                1.8042080500337247e+32,
                1.127629910378496e+32,
                8.993372290290595e+31,
                1.438940881757787e+32,
                1.4389399146171313e+32,
                8.993401304510266e+31,
                5.188482823302229e+31,
                8.301574644993008e+31,
                8.301579480696287e+31,
                5.188481372591245e+31,
            ]))
        });
    });
});
