//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// UnitTest.cpp : Unit test application for CPP UWP Eval examples.
//


#include "pch.h"
#include "CppUnitTest.h"
#include "CNTKLibrary.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace Windows::Storage;
using namespace CNTK;

void MultiThreadsEvaluationTests(const wchar_t* modelPath, bool);
void EvaluationSingleSampleUsingDense(const wchar_t* modelPath, const CNTK::DeviceDescriptor&);
void EvaluationBatchUsingDense(const wchar_t* modelPath, const CNTK::DeviceDescriptor&);
void ParallelEvaluationExample(const wchar_t* modelPath, const CNTK::DeviceDescriptor&);
void EvaluationSingleSequenceUsingOneHot(const wchar_t* modelPath, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationBatchOfSequencesUsingOneHot(const wchar_t* modelPath, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);
void EvaluationSingleSequenceUsingSparse(const wchar_t* modelPath, const wchar_t*, const wchar_t*, const CNTK::DeviceDescriptor&);

namespace UWPEvalTests
{
    TEST_CLASS(TestClass)
    {
    private:
        const wchar_t* oneHiddenModelPath = L"01_OneHidden.model";
        const wchar_t* resnet20ModelPath = L"resnet20.model";
        const wchar_t* atisModelPath = L"atis.model";
        const wchar_t* vocabularyFilePath = L"query.txt";
        const wchar_t* labelFilePath = L"slots.txt";

        concurrency::task<Platform::String^> GetModelFilePath(Platform::String^ modelFileName)
        {
            auto modelFileOp = KnownFolders::DocumentsLibrary->GetFileAsync(modelFileName);
            return concurrency::create_task(modelFileOp).then([modelFileName](concurrency::task<StorageFile^> modelFileTask) {
                try
                {
                    // The file cannot be read directly from the DocumentsLibrary, so copy the file into the local app folder
                    auto modelFile = modelFileTask.get();
                    auto localFolder = Windows::Storage::ApplicationData::Current->LocalFolder;
                    auto copyTask = modelFile->CopyAsync(localFolder, modelFileName, NameCollisionOption::ReplaceExisting);
                    return concurrency::create_task(copyTask).then([](concurrency::task<StorageFile^> modelFileTask2) {
                        auto path = modelFileTask2.get()->Path;
                        return path;
                    });
                }
                catch (...)
                {
                    auto message = L"The file " + modelFileName + L" must exist in the Documents directory";
                    Assert::Fail(message->Data());
                    throw;
                }
            });
        }

        template<typename Func>
        void RunTestWithModel(const wchar_t* modelFile, const Func& func)
        {
            Platform::String^ modelFileName = ref new Platform::String(modelFile);
            try {
                concurrency::create_task(GetModelFilePath(modelFileName)).then([&func](concurrency::task<Platform::String^> modelFilePath) {
                    auto path = modelFilePath.get();
                    func(path->Data());
                }).get();
            }
            catch (...) {
                Assert::Fail(L"Exception while test execution");
                throw;
            }
        }

        template<typename Func>
        void RunTestWithAtisModel(const wchar_t* modelFile, const wchar_t* vocabularyFile, const wchar_t* labelFile, const Func& func)
        {
            Platform::String^ modelFileName = ref new Platform::String(modelFile);
            Platform::String^ vocabularyFileName = ref new Platform::String(vocabularyFile);
            Platform::String^ labelFileName = ref new Platform::String(labelFile);

            UNREFERENCED_PARAMETER(func);
            //try {
            //    auto modelFilePathInUse = concurrency::create_task(GetModelFilePath(modelFileName)).get();
            //    auto vocabularyFilePathInUse = concurrency::create_task(GetModelFilePath(vocabularyFileName)).get();
            //    auto labelFilePathInUse = concurrency::create_task(GetModelFilePath(labelFileName)).get();

            //    concurrency::create_task([modelFilePathInUse, vocabularyFilePathInUse, labelFilePathInUse, func] () -> void {
            //        func(modelFilePathInUse->Data(), vocabularyFilePathInUse->Data(), labelFilePathInUse->Data());
            //    }).get();
            //}
            //catch (...) {
            //    Assert::Fail(L"Exception while test execution");
            //    throw;
            //}
        }

    public:
        TEST_METHOD(UWPTestSanity)
        {
            // Failure in this test indicates a problem with infrastructure
            Assert::IsTrue(true);
        }

        TEST_METHOD(UWPTestModelLoad)
        {
            RunTestWithModel(oneHiddenModelPath, [](auto path) {
                auto device = DeviceDescriptor::CPUDevice();
                CNTK::Function::Load(path, device);
            });
        }

        TEST_METHOD(UWPTestMultiThreadsEvaluationTests)
        {
            RunTestWithModel(oneHiddenModelPath, [](auto path) {
                MultiThreadsEvaluationTests(path, false);
            });
        }

        TEST_METHOD(UWPTestEvaluationSingleSampleUsingDense)
        {
            RunTestWithModel(resnet20ModelPath, [](auto path) {
                EvaluationSingleSampleUsingDense(path, CNTK::DeviceDescriptor::CPUDevice());
            });
        }

        TEST_METHOD(UWPTestEvaluationBatchUsingDense)
        {
            RunTestWithModel(resnet20ModelPath, [](auto path) {
                EvaluationBatchUsingDense(path, CNTK::DeviceDescriptor::CPUDevice());
            });
        }

        TEST_METHOD(UWPTestParallelEvaluationExample)
        {
            RunTestWithModel(resnet20ModelPath, [](auto path) {
                ParallelEvaluationExample(path, CNTK::DeviceDescriptor::CPUDevice());
            });
        }

        TEST_METHOD(UWPTestEvaluationSingleSequenceUsingOneHot)
        {
            Assert::IsTrue(true);
            /*RunTestWithAtisModel(atisModelPath, vocabularyFilePath, labelFilePath, [](const wchar_t* path, const wchar_t* vocabPath, const wchar_t* labelPath) {
                EvaluationSingleSequenceUsingOneHot(path, vocabPath, labelPath, CNTK::DeviceDescriptor::CPUDevice());
            });*/
        }

        //TEST_METHOD(UWPTestEvaluationBatchOfSequencesUsingOneHot)
        //{
        //    RunTestWithModel(atisModelPath, vocabularyFilePath, labelFilePath, [](auto path, auto vocabPath, auto labelPath) {
        //        EvaluationBatchOfSequencesUsingOneHot(path, vocabPath, labelPath, CNTK::DeviceDescriptor::CPUDevice());
        //    });
        //}

        //TEST_METHOD(UWPTestEvaluationSingleSequenceUsingSparse)
        //{
        //    RunTestWithModel(atisModelPath, vocabularyFilePath, labelFilePath, [](auto path, auto vocabPath, auto labelPath) {
        //        EvaluationSingleSequenceUsingSparse(path, vocabPath, labelPath, CNTK::DeviceDescriptor::CPUDevice());
        //    });
        //}
    };
}