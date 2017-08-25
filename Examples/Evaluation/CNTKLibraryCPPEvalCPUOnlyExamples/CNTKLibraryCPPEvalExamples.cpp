//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibraryCPPEvalExamples.cpp : Sample application shows how to evaluate a model using CNTK V2 API.
//

#include "CNTKLibrary.h"

using namespace CNTK;

template <typename ElementType>
void PrintOutput(size_t, std::vector<std::vector<ElementType>>);

/// <summary>
/// The example shows
/// - how to load model.
/// - how to prepare input data for a single sample.
/// - how to prepare input and output data map.
/// - how to evaluate a model.
/// - how to retrieve evaluation result and retrieve output data in dense format.
/// Note: It uses the model trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
/// Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
/// The pre-trained model file must be in the output directory.
/// </summary>
void EvaluationSingleSampleUsingDense(const DeviceDescriptor& device)
{
    printf("\n===== Evaluate single sample using dense format.\n");

    // Load the model.
    // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
    // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
    const wchar_t* modelFileName = L"resnet20.dnn";
    FunctionPtr modelFunc = Function::Load(modelFileName, device);

    // Get input variable. The model has only one single input.
    Variable inputVar = modelFunc->Arguments()[0];

    // The model has only one output.
    // If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
    Variable outputVar = modelFunc->Output();

    // Prepare input data.
    // For evaluating an image, you first need to perform some image preprocessing to make sure that the input image has the correct size and layout
    // that match the model inputs.
    // Please note that the model used by this example expects the CHW image layout.
    // inputVar.Shape[0] is image width, inputVar.Shape[1] is image height, and inputVar.Shape[2] is channels.
    // For simplicity and avoiding external dependencies, we skip the preprocessing step here, and just use some artificially created data as input.
    std::vector<float> inputData(inputVar.Shape().TotalSize());
    for (size_t i = 0; i < inputData.size(); ++i)
    {
        inputData[i] = static_cast<float>(i % 255);
    }

    // Create input value and input data map
    ValuePtr inputVal = Value::CreateBatch(inputVar.Shape(), inputData, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputVar, inputVal } };

    // Create output data map. Using null as Value to indicate using system allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputVar, nullptr } };

    // Start evaluation on the device
    modelFunc->Evaluate(inputDataMap, outputDataMap, device);

    // Get evaluate result as dense output
    ValuePtr outputVal = outputDataMap[outputVar];
    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);

    PrintOutput<float>(outputVar.Shape().TotalSize(), outputData);
}


/// <summary>
/// The example shows
/// - how to load model.
/// - how to prepare input data for a batch of samples.
/// - how to prepare input and output data map.
/// - how to evaluate a model.
/// - how to retrieve evaluation result and retrieve output data in dense format.
/// Note: It uses the model trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
/// Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
/// The pre-trained model file must be in the output directory.
/// </summary>
void EvaluationBatchUsingDense(const DeviceDescriptor& device)
{
    printf("\n===== Evaluate batch of samples using dense format.\n");

    // The number of samples in the batch.
    size_t sampleCount = 3;

    // Load the model.
    // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
    // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
    const wchar_t* modelFileName = L"resnet20.dnn";
    FunctionPtr modelFunc = Function::Load(modelFileName, device);

    // Get input variable. The model has only one single input.
    Variable inputVar = modelFunc->Arguments()[0];

    // The model has only one output.
    // If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
    Variable outputVar = modelFunc->Output();

    // Prepare input data.
    // For evaluating an image, you first need to perform some image preprocessing to make sure that the input image has the correct size and layout
    // that match the model inputs.
    // Please note that the model used by this example expects the CHW image layout.
    // inputVar.Shape[0] is image width, inputVar.Shape[1] is image height, and inputVar.Shape[2] is channels.
    // For simplicity and avoiding external dependencies, we skip the preprocessing step here, and just use some artificially created data as input.
    std::vector<float> inputData(inputVar.Shape().TotalSize() * sampleCount);
    for (size_t i = 0; i < inputData.size(); ++i)
    {
        inputData[i] = static_cast<float>(i % 255);
    }

    // Create input value and input data map.
    ValuePtr inputVal = Value::CreateBatch(inputVar.Shape(), inputData, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputVar, inputVal } };

    // Create output data map. Using null as Value to indicate using system allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputVar, nullptr } };

    // Start evaluation on the device
    modelFunc->Evaluate(inputDataMap, outputDataMap, device);

    // Get evaluate result as dense output
    ValuePtr outputVal = outputDataMap[outputVar];
    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);

    PrintOutput<float>(outputVar.Shape().TotalSize(), outputData);
}

void RunEvaluationOnSingleSample(FunctionPtr, const DeviceDescriptor&);

/// <summary>
/// The example shows
/// - how to evaluate multiple sample requests in parallel.
/// Note: It uses the model trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
/// Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
/// The pre-trained model file must be in the output directory.
/// </summary>
void ParallelEvaluationExample(const DeviceDescriptor& device)
{
    printf("\n===== Evaluate multiple requests in parallel.\n");

    size_t threadCount = 3;

    // Load the model.
    // The model resnet20.dnn is trained by <CNTK>/Examples/Image/Classification/ResNet/Python/TrainResNet_CIFAR10.py
    // Please see README.md in <CNTK>/Examples/Image/Classification/ResNet about how to train the model.
    const wchar_t* modelFileName = L"resnet20.dnn";
    FunctionPtr modelFunc = Function::Load(modelFileName, device);

    // Run evaluation in parallel.
    std::vector<std::thread> threadList(threadCount);
    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th] = std::thread(RunEvaluationOnSingleSample, modelFunc->Clone(ParameterCloningMethod::Share), device);
    }

    for (int th = 0; th < threadCount; ++th)
    {
        threadList[th].join();
        printf("thread %d joined.\n", th);
    }
}

void RunEvaluationOnSingleSample(FunctionPtr evalInstance, const DeviceDescriptor& device)
{
    // Get input variable. The model has only one single input.
    Variable inputVar = evalInstance->Arguments()[0];

    // The model has only one output.
    // If the model has more than one output, use modelFunc->Outputs to get the list of output variables.
    Variable outputVar = evalInstance->Output();

    // Prepare input data.
    // For evaluating an image, you first need to perform some image preprocessing to make sure that the input image has the correct size and layout
    // that match the model inputs.
    // Please note that the model used by this example expects the CHW image layout.
    // inputVar.Shape[0] is image width, inputVar.Shape[1] is image height, and inputVar.Shape[2] is channels.
    // For simplicity and avoiding external dependencies, we skip the preprocessing step here, and just use some artificially created data as input.
    std::vector<float> inputData(inputVar.Shape().TotalSize());
    for (size_t i = 0; i < inputData.size(); ++i)
    {
        inputData[i] = static_cast<float>(i % 255);
    }

    // Create input value and input data map
    ValuePtr inputVal = Value::CreateBatch(inputVar.Shape(), inputData, device);
    std::unordered_map<Variable, ValuePtr> inputDataMap = { { inputVar, inputVal } };

    // Create output data map. Using null as Value to indicate using system allocated memory.
    // Alternatively, create a Value object and add it to the data map.
    std::unordered_map<Variable, ValuePtr> outputDataMap = { { outputVar, nullptr } };

    // Start evaluation on the device
    evalInstance->Evaluate(inputDataMap, outputDataMap, device);

    // Get evaluate result as dense output
    ValuePtr outputVal = outputDataMap[outputVar];
    std::vector<std::vector<float>> outputData;
    outputVal->CopyVariableValueTo(outputVar, outputData);
}

/// <summary>
/// Print out the evalaution results.
/// </summary>
template <typename ElementType>
void PrintOutput(size_t sampleSize, std::vector<std::vector<ElementType>> outputBuffer)
{
    printf("The batch contains %d sequences.\n", (int)outputBuffer.size());
    for(size_t seqNo = 0; seqNo < outputBuffer.size(); seqNo++)
    {
        auto seq = outputBuffer[seqNo];
        if (seq.size() % sampleSize != 0)
            throw("The number of elements in the sequence is not a multiple of sample size");

        printf("Sequence %d contains %d samples.\n", (int)seqNo, (int)(seq.size() / sampleSize));
        size_t sampleNo = 0;
        for(size_t i = 0; i < seq.size(); )
        {
            if (i % sampleSize == 0)
                printf("    sample %d: ", (int)sampleNo);
            printf("%f", seq[i++]);
            if (i % sampleSize == 0)
            {
                printf(".\n");
                sampleNo++;
            }
            else
                printf(", ");
        }
    }
}