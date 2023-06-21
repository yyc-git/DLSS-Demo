import { loadFeature, defineFeature } from 'jest-cucumber';
import { sizeOfShape } from '../../src/common/utils';
import { build, createState, _prepareWeightForZeroUpsampling, _remap, _zeroUpsampling } from '../../src/nsrr';
import { computeWithNoInput } from '../utils/ComputeUtils';
import { buildTensor } from '../utils/TensorUtils';

const feature = loadFeature('./test/features/nsrr.feature');

defineFeature(feature, test => {
    let state
    let context, builder
    let results

    // function _buildWeight(dimensions, value) {
    //   return builder.constant({ type: "float32", dimensions: dimensions },
    //     new Float32Array(sizeOfShape(dimensions)).fill(value)
    //   )
    // }
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
            state = createState()
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
                    _prepareWeightForZeroUpsampling(state, channel).weightForZeroUpsampling,
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
            state = createState()
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
});
