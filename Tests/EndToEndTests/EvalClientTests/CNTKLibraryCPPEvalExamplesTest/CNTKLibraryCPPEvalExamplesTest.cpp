//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCPPEvalExamplesTest.cpp : Test application for CNTKLibraryCPPEvalExamples, both for CPUOnly and GPU.
//

#include <stdio.h>

#include <CNTKLibrary.h>

void MultiThreadsEvaluationTests(bool);
void EvaluationSingleSampleUsingDense(const CNTK::DeviceDescriptor&);
void EvaluationBatchUsingDense(const CNTK::DeviceDescriptor&);
void ParallelEvaluationExample(const CNTK::DeviceDescriptor&);
void EvaluationSingleSequenceUsingOneHot(const CNTK::DeviceDescriptor&);
void EvaluationBatchOfSequencesUsingOneHot(const CNTK::DeviceDescriptor&);
void EvaluationSingleSequenceUsingSparse(const CNTK::DeviceDescriptor&);
bool ShouldRunOnCpu();
bool ShouldRunOnGpu();

int main()
{
    if (ShouldRunOnGpu())
    {
        fprintf(stderr, "\n##### Test CPPEval samples on GPU device. #####\n");
        EvaluationSingleSampleUsingDense(CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationBatchUsingDense(CNTK::DeviceDescriptor::GPUDevice(0));
        ParallelEvaluationExample(CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationSingleSequenceUsingOneHot(CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationBatchOfSequencesUsingOneHot(CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationSingleSequenceUsingSparse(CNTK::DeviceDescriptor::GPUDevice(0));

        fprintf(stderr, "\n##### Test MultiThreadsEvaluation on GPU device. #####\n");
        MultiThreadsEvaluationTests(true);
    }

    if (ShouldRunOnCpu())
    {
        fprintf(stderr, "\n##### Test CPPEval samples on CPU device. #####\n");
        EvaluationSingleSampleUsingDense(CNTK::DeviceDescriptor::CPUDevice());
        EvaluationBatchUsingDense(CNTK::DeviceDescriptor::CPUDevice());
        ParallelEvaluationExample(CNTK::DeviceDescriptor::CPUDevice());
        EvaluationSingleSequenceUsingOneHot(CNTK::DeviceDescriptor::CPUDevice());
        EvaluationBatchOfSequencesUsingOneHot(CNTK::DeviceDescriptor::CPUDevice());
        EvaluationSingleSequenceUsingSparse(CNTK::DeviceDescriptor::CPUDevice());

        fprintf(stderr, "\n##### Test MultiThreadsEvaluation CPU device. #####\n");
        MultiThreadsEvaluationTests(false);
    }

    fprintf(stderr, "Evaluation complete.\n");
    fflush(stderr);
}
