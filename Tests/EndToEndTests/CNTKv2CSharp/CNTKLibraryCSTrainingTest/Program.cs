//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Program.cs : Tests of CNTK Library C# model training examples.
//
using CNTK;
using CNTK.CSTrainingExamples;
using System;

namespace CNTK.CNTKLibraryCSTrainingTest
{
    class Program
    {
        static void Main(string[] args)
        {
            // Todo: move to a separate unit test.
            Console.WriteLine("Test CNTKLibraryCSTrainingExamples");
#if CPUONLY
            Console.WriteLine("======== Train model on CPU using CPUOnly build ========");
#else
            Console.WriteLine("======== Train model on CPU using GPU build ========");
#endif
            if (ShouldRunOnCpu())
            {
                Console.WriteLine(" ====== Train model on CPU =====");
                var device = DeviceDescriptor.CPUDevice;

                SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier(device);

                if (args.Length > 0 && args[0] == "RunExamples")
                {
                    RunExamples(device);
                }
            }

            if (ShouldRunOnGpu())
            {
                Console.WriteLine(" ====== Train model on GPU =====");
                var device = DeviceDescriptor.GPUDevice(0);

                SimpleFeedForwardClassifierTest.TrainSimpleFeedForwardClassifier(device);

                if (args.Length > 0 && args[0] == "RunExamples")
                {
                    RunExamples(device);
                }
            }

            Console.WriteLine("======== Train completes. ========");
        }

        static bool ShouldRunOnGpu()
        {
#if CPUONLY
            return false;
#else
            string testDeviceSetting = Environment.GetEnvironmentVariable("TEST_DEVICE");

            return (string.IsNullOrEmpty(testDeviceSetting) || string.Equals(testDeviceSetting.ToLower(), "gpu"));
#endif
        }

        static bool ShouldRunOnCpu()
        {
            string testDeviceSetting = Environment.GetEnvironmentVariable("TEST_DEVICE");

            return (string.IsNullOrEmpty(testDeviceSetting) || string.Equals(testDeviceSetting.ToLower(), "cpu"));
        }

        static void RunExamples(DeviceDescriptor device)
        {
            Console.WriteLine($"======== runing MNISTClassifierTest.TrainAndEvaluate using {device.Type} with logistic classifier ========");
            MNISTClassifier.TrainAndEvaluate(device, false, true);

            Console.WriteLine($"======== runing MNISTClassifierTest.TrainAndEvaluate using {device.Type} with convolution classifier ========");
            MNISTClassifier.TrainAndEvaluate(device, true, true);

            if (device.Type == DeviceKind.GPU)
            {
                Console.WriteLine($"======== runing CifarResNet.TrainAndEvaluate using {device.Type} ========");
                CifarResNetClassifier.TrainAndEvaluate(device, true);
            }

            if (device.Type == DeviceKind.GPU)
            {
                Console.WriteLine($"======== runing TransferLearning.TrainAndEvaluateWithFlowerData using {device.Type} ========");
                TransferLearning.TrainAndEvaluateWithFlowerData(device, true);

                Console.WriteLine($"======== runing TransferLearning.TrainAndEvaluateWithAnimalData using {device.Type} ========");
                TransferLearning.TrainAndEvaluateWithAnimalData(device, true);
            }

            Console.WriteLine($"======== runing LSTMSequenceClassifier.Train using {device.Type} ========");
            LSTMSequenceClassifier.Train(device, true);
        }
    }
}
