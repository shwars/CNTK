using System.Collections.Generic;
using System.IO;

namespace CNTK.CSTrainingExamples
{
    public class MNISTClassifier
    {
        public static string ImageDataFolder = "C:/cntk/Tests/EndToEndTests/Image/Data/";

        public static void TrainAndEvaluate(DeviceDescriptor device, bool useConvolution, bool forceRetrain)
        {
            Function classifierOutput;
            int[] imageDim = useConvolution ? new int[] { 28, 28, 1 } : new int[] { 784 };
            var input = CNTKLib.InputVariable(imageDim, DataType.Float, "features");
            int imageSize = 28 * 28;
            int numClasses = 10;

            var featureStreamName = "features";
            var labelsStreamName = "labels";
            IList<StreamConfiguration> streamConfigurations = new StreamConfiguration[]
                { new StreamConfiguration(featureStreamName, imageSize), new StreamConfiguration(labelsStreamName, numClasses) };

            string modelFile = useConvolution ? "MNISTConvolution.model" : "MNISTLogisticRegression.model";
            if (File.Exists(modelFile) && !forceRetrain)
            {
                var minibatchSourceExistModel = MinibatchSource.TextFormatMinibatchSource(
                    Path.Combine(ImageDataFolder, "Test_cntk_text.txt"), streamConfigurations);
                TestHelper.ValidateModelWithMinibatchSource(modelFile, minibatchSourceExistModel,
                                    imageDim, numClasses, "features", "labels", "classifierOutput", device);
                return;
            }

            if (useConvolution)
            {

                var scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float>(0.00390625f, device), input);
                classifierOutput = CreateConvolutionalNeuralNetwork(scaledInput, numClasses, device);
            }
            else
            {
                int hiddenLayerDim = 200;
                var scaledInput = CNTKLib.ElementTimes(Constant.Scalar<float>(0.00390625f, device), input);
                classifierOutput = CreateLogisticClassifier(device, numClasses, hiddenLayerDim, scaledInput);
            }

            var labels = CNTKLib.InputVariable(new int[] { numClasses }, DataType.Float, "labels");
            var trainingLoss = CNTKLib.CrossEntropyWithSoftmax(new Variable(classifierOutput), labels, "lossFunction");
            var prediction = CNTKLib.ClassificationError(new Variable(classifierOutput), labels, "classificationError");

            const uint minibatchSize = 64;
            const uint numSamplesPerSweep = 6000;
            const uint numSweepsToTrainWith = 2;
            const uint numMinibatchesToTrain = (numSamplesPerSweep * numSweepsToTrainWith) / minibatchSize;
            
            var minibatchSource = MinibatchSource.TextFormatMinibatchSource(
                Path.Combine(ImageDataFolder, "Train_cntk_text.txt"), streamConfigurations);

            var featureStreamInfo = minibatchSource.StreamInfo(featureStreamName);
            var labelStreamInfo = minibatchSource.StreamInfo(labelsStreamName);

            CNTK.TrainingParameterScheduleDouble learningRatePerSample = new CNTK.TrainingParameterScheduleDouble(
                0.003125, TrainingParameterScheduleDouble.UnitType.Sample);

            IList<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(classifierOutput.Parameters(), learningRatePerSample) };
            var trainer = Trainer.CreateTrainer(classifierOutput, trainingLoss, prediction, parameterLearners);

            int outputFrequencyInMinibatches = 20;
            for (int i = 0; i < numMinibatchesToTrain; ++i)
            {
                var minibatchData = minibatchSource.GetNextMinibatch(minibatchSize, device);
                var arguments = new Dictionary<Variable, MinibatchData>
                {
                    { input, minibatchData[featureStreamInfo] },
                    { labels, minibatchData[labelStreamInfo] }
                };
                trainer.TrainMinibatch(arguments, device);
                TestHelper.PrintTrainingProgress(trainer, i, outputFrequencyInMinibatches);
            }

            classifierOutput.Save(modelFile);

            var minibatchSourceNewModel = MinibatchSource.TextFormatMinibatchSource(
                Path.Combine(ImageDataFolder, "Test_cntk_text.txt"), streamConfigurations);
            TestHelper.ValidateModelWithMinibatchSource(modelFile, minibatchSourceNewModel,
                                imageDim, numClasses, "features", "labels", "classifierOutput", device);
        }

        private static Function CreateLogisticClassifier(DeviceDescriptor device, int numOutputClasses, int hiddenLayerDim, Function scaledInput)
        {
            Function toSigmoid = TestHelper.FullyConnectedLinearLayer(new Variable(scaledInput), hiddenLayerDim, device, "");
            var classifierOutput = CNTKLib.Sigmoid(new Variable(toSigmoid), "");


            var outputTimesParam = new Parameter(NDArrayView.RandomUniform<float>(
                new int[] { numOutputClasses, hiddenLayerDim }, -0.05, 0.05, 1, device));
            var outputBiasParam = new Parameter(NDArrayView.RandomUniform<float>(new int[] { numOutputClasses }, -0.05, 0.05, 1, device));
            classifierOutput = CNTKLib.Plus(outputBiasParam, new Variable(CNTKLib.Times(outputTimesParam, new Variable(classifierOutput))), "classifierOutput");
            return classifierOutput;
        }

        static Function CreateConvolutionalNeuralNetwork(Variable features, int outDims, DeviceDescriptor device)
        {
            int kernelWidth1 = 3, kernelHeight1 = 3, numInputChannels1 = 1, outFeatureMapCount1 = 4;
            int hStride1 = 2, vStride1 = 2;
            int poolingWindowWidth1 = 3, poolingWindowHeight1 = 3;

            // // 28x28x1 -> 14x14x4
            Function pooling1 = ConvolutionWithMaxPooling(features, device, kernelWidth1, kernelHeight1,
                numInputChannels1, outFeatureMapCount1, hStride1, vStride1, poolingWindowWidth1, poolingWindowHeight1);

            // return TestHelper.Dense(pooling1, outDims, device, Activation.None, "classifierOutput");

            // 14x14x8 -> 7x7x8
            int kernelWidth2 = 3, kernelHeight2 = 3, numInputChannels2 = outFeatureMapCount1, outFeatureMapCount2 = 8;
            int hStride2 = 2, vStride2 = 2;
            int poolingWindowWidth2 = 3, poolingWindowHeight2 = 3;

            Function pooling2 = ConvolutionWithMaxPooling(pooling1, device, kernelWidth2, kernelHeight2,
                numInputChannels2, outFeatureMapCount2, hStride2, vStride2, poolingWindowWidth2, poolingWindowHeight2);

            Function denseLayer = TestHelper.Dense(pooling2, outDims, device, Activation.None, "classifierOutput");
            TestHelper.PrintOutputDims(denseLayer, "denseLayer");
            return denseLayer;
        }

        private static Function ConvolutionWithMaxPooling(Variable features, DeviceDescriptor device, 
            int kernelWidth, int kernelHeight, int numInputChannels, int outFeatureMapCount, 
            int hStride, int vStride, int poolingWindowWidth, int poolingWindowHeight)
        {
            double convWScale = 0.26;
            var convParams = new Parameter(new int[] { kernelWidth, kernelHeight, numInputChannels, outFeatureMapCount }, DataType.Float,
                CNTKLib.GlorotUniformInitializer(convWScale, -1, 2), device);
            Function convFunction = CNTKLib.ReLU(CNTKLib.Convolution(convParams, features, new int[] { 1, 1, numInputChannels } /* strides */));

            TestHelper.PrintOutputDims(convFunction, "convFunction1");
            Function pooling = CNTKLib.Pooling(convFunction, PoolingType.Max,
                new int[] { poolingWindowWidth, poolingWindowHeight }, new int[] { hStride, vStride }, new bool[] { true });
            TestHelper.PrintOutputDims(pooling, "pooling");
            return pooling;
        }
    }
}
