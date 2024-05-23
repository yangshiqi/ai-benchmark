# -*- coding: utf-8 -*-
# Copyright 2019-2020 by Andrey Ignatov. All Rights Reserved.

from __future__ import print_function
from os import path


class SubTest:

    def __init__(self, batch_size, input_dimensions, output_dimensions, iterations, min_passes, max_duration, ref_time,
                 loss_function=None, optimizer=None, learning_rate=None):

        self.batch_size = batch_size
        self.input_dimensions = input_dimensions
        self.output_dimensions = output_dimensions
        self.iterations = iterations
        self.min_passes = min_passes
        self.max_duration = max_duration
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.ref_time = ref_time

    def getInputDims(self):
        inputDims = [self.batch_size]
        inputDims.extend(self.input_dimensions)
        return inputDims

    def getOutputDims(self):
        outputDims = [self.batch_size]
        outputDims.extend(self.output_dimensions)
        return outputDims


class Test:

    def __init__(self, test_id, test_type, model, model_src, use_src, tests_training, tests_inference, tests_micro):

        self.id = test_id
        self.type = test_type
        self.model = model
        self.model_src = path.join(path.dirname(__file__), "models/" + model_src)
        self.use_src = use_src
        self.training = tests_training
        self.inference = tests_inference
        self.micro = tests_micro


class TestConstructor:

    def getTests(self, batch_mul=1):

        benchmark_tests = [

            Test(test_id=5, test_type="classification", model="ResNet-V2-50", use_src=False,
                 model_src="resnet_v2_50.meta",
                 tests_training=[SubTest(10*batch_mul, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=172)],
                 tests_inference=[SubTest(10, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=48)],
                 tests_micro=[SubTest(1, [346, 346, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=10)]),

            Test(test_id=6, test_type="classification", model="ResNet-V2-152", use_src=False,
                 model_src="resnet_v2_152.meta",
                 tests_training=[SubTest(10*batch_mul, [256, 256, 3], [1001], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=265)],
                 tests_inference=[SubTest(10, [256, 256, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=60)],
                 tests_micro=[SubTest(1, [256, 256, 3], [1001], 22, min_passes=5, max_duration=30, ref_time=25)]),

            Test(test_id=7, test_type="classification", model="VGG-16", use_src=False,
                 model_src="vgg_16.meta",
                 tests_training=[SubTest(2*batch_mul, [224, 224, 3], [1000], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=190)],
                 tests_inference=[SubTest(20, [224, 224, 3], [1000], 22, min_passes=5, max_duration=30, ref_time=110)],
                 tests_micro=[SubTest(1, [224, 224, 3], [1000], 22, min_passes=5, max_duration=30, ref_time=56)]),

            Test(test_id=16, test_type="segmentation", model="DeepLab", model_src="deeplab.meta", use_src=False,
                 tests_training=[SubTest(1*batch_mul, [384, 384, 3], [48, 48, 3], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=191)],
                 tests_inference=[
                     SubTest(2, [512, 512, 3], [64, 64, 3], 22, min_passes=5, max_duration=30, ref_time=125)],
                 tests_micro=[
                     SubTest(1, [512, 512, 3], [512, 512, 3], 22, min_passes=5, max_duration=30, ref_time=67)]),

            Test(test_id=18, test_type="nlp", model="LSTM-Sentiment", model_src="lstm.meta", use_src=True,
                 tests_training=[SubTest(10*batch_mul, [1024, 300], [2], 22, min_passes=5, max_duration=30,
                                         loss_function="MSE", optimizer="Adam", learning_rate=1e-4, ref_time=728)],
                 tests_inference=[SubTest(100, [1024, 300], [2], 22, min_passes=5, max_duration=30, ref_time=547)],
                 tests_micro=[])
        ]

        return benchmark_tests
