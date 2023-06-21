Feature: NSRR
    As a NSRR algorithm
    I want to supersample with webnn api
    So that I can get upsampled scene image data

    Scenario: zero upsampling
        Given create context
        And set backend to cpu
        And create builder
        And create state with fake all_features
        And zero upsampling
        And build
        When compute with no input
        Then get zero upsampling data
