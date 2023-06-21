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
