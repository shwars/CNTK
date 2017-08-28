//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCPPEvalExamplesTest.cpp : Test application for CNTKLibraryCPPEvalExamples, both for CPUOnly and GPU.
//

#include <stdio.h>

#include <CNTKLibrary.h>

void MultiThreadsEvaluationTests(const wchar_t* modelPath, bool);
void EvaluationSingleSampleUsingDense(const wchar_t* modelPath, const CNTK::DeviceDescriptor&);
void EvaluationBatchUsingDense(const wchar_t* modelPath, const CNTK::DeviceDescriptor&);
void ParallelEvaluationExample(const wchar_t* modelPath, const CNTK::DeviceDescriptor&);
void EvaluationSingleSequenceUsingOneHot(const wchar_t* modelPath, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationBatchOfSequencesUsingOneHot(const wchar_t* modelPath, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationSingleSequenceUsingSparse(const wchar_t* modelPath, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);
bool ShouldRunOnCpu();
bool ShouldRunOnGpu();

int main()
{
    const wchar_t* oneHiddenModelPath = L"01_OneHidden.model";
    const wchar_t* resnet20ModelPath = L"resnet20.dnn";
    const wchar_t* atisModelPath = L"atis.dnn";
    const wchar_t* vocabularyFilePath = L"query.wl";
    const wchar_t* labelFilePath = L"slots.wl";

    if (ShouldRunOnGpu())
    {
        fprintf(stderr, "\n##### Test CPPEval samples on GPU device. #####\n");
        EvaluationSingleSampleUsingDense(resnet20ModelPath, CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationBatchUsingDense(resnet20ModelPath, CNTK::DeviceDescriptor::GPUDevice(0));
        ParallelEvaluationExample(resnet20ModelPath, CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationSingleSequenceUsingOneHot(atisModelPath, vocabularyFilePath, labelFilePath, CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationBatchOfSequencesUsingOneHot(atisModelPath, vocabularyFilePath, labelFilePath, CNTK::DeviceDescriptor::GPUDevice(0));
        EvaluationSingleSequenceUsingSparse(atisModelPath, vocabularyFilePath, labelFilePath, CNTK::DeviceDescriptor::GPUDevice(0));

        fprintf(stderr, "\n##### Test MultiThreadsEvaluation on GPU device. #####\n");
        MultiThreadsEvaluationTests(oneHiddenModelPath, true);
    }

    if (ShouldRunOnCpu())
    {
        fprintf(stderr, "\n##### Test CPPEval samples on CPU device. #####\n");
        EvaluationSingleSampleUsingDense(resnet20ModelPath, CNTK::DeviceDescriptor::CPUDevice());
        EvaluationBatchUsingDense(resnet20ModelPath, CNTK::DeviceDescriptor::CPUDevice());
        ParallelEvaluationExample(resnet20ModelPath, CNTK::DeviceDescriptor::CPUDevice());
        EvaluationSingleSequenceUsingOneHot(atisModelPath, vocabularyFilePath, labelFilePath, CNTK::DeviceDescriptor::CPUDevice());
        EvaluationBatchOfSequencesUsingOneHot(atisModelPath, vocabularyFilePath, labelFilePath, CNTK::DeviceDescriptor::CPUDevice());
        EvaluationSingleSequenceUsingSparse(atisModelPath, vocabularyFilePath, labelFilePath, CNTK::DeviceDescriptor::CPUDevice());

        fprintf(stderr, "\n##### Test MultiThreadsEvaluation CPU device. #####\n");
        MultiThreadsEvaluationTests(oneHiddenModelPath, false);
    }

    fprintf(stderr, "Evaluation complete.\n");
    fflush(stderr);
}
