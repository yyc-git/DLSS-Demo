import { loadFeature, defineFeature } from 'jest-cucumber';
import { sizeOfShape } from '../../src/common/utils';
import { build, createState, _prepareWeightForZeroUpsampling, _zeroUpsampling } from '../../src/nsrr';
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


        and('create builder', async () => {
            builder = new MLGraphBuilder(context)
        })

        and('create state with fake all_features', async () => {
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

        and('zero upsampling', () => {
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

        when('compute with no input', async () => {
            let outputBuffer = new Float32Array(sizeOfShape([n, channel, height * 4, width * 4]));

            let inputs = {}
            let outputs = { 'output': outputBuffer }
            results = await state.context.compute(state.graph, inputs, outputs);
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
});
