Feature: NSRR
    As a NSRR algorithm
    I want to supersample with webnn api
    So that I can get upsampled scene image data

    Background: prepare
        Given create context
        And set backend to cpu
        And create builder

    Scenario: zero upsampling
        Given create state with fake all_features
        When zero upsampling
        And build
        And compute with no input
        Then get zero upsampling data

    Scenario: remap
        Given prepare tensor in range [-1,1]
        And create state
        When remap tensor to range [0,10]
        And build
        And compute with no input
        Then get remaped data in range [0,10]

    Scenario: create compute graph of whole
        Given prepare fake input: view_tensor, depth_tensor
        And create state
        And prepare weightForZeroUpsampling
        When create compute graph of input
        And create compute graph of feature extract
        And create compute graph of zero upsampling
        And create compute graph of feature reweighting
        And create compute graph of reconstruction
        And build
        And compute with input
        Then get correct data
