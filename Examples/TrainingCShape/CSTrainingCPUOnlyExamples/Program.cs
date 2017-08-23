using CNTK;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CSTrainingExamples
{
    class Program
    {
        static void Main(string[] args)
        {
            var device = DeviceDescriptor.CPUDevice;
            Console.WriteLine("======== runing MNISTClassifierTest.TrainAndEvaluate using CPU with logistic classifier ========");
            MNISTClassifier.TrainAndEvaluate(device, false, true);

            Console.WriteLine("======== runing MNISTClassifierTest.TrainAndEvaluate using CPU with convolution classifier ========");
            MNISTClassifier.TrainAndEvaluate(device, true, true);

            Console.WriteLine("======== runing CifarResNet.TrainAndEvaluate using CPU ========");
            CifarResNetClassifier.TrainAndEvaluate(device, true);

            Console.WriteLine("======== runing TransferLearning.TrainAndEvaluateWithFlowerData using CPU ========");
            TransferLearning.TrainAndEvaluateWithFlowerData(device, true);

            Console.WriteLine("======== runing TransferLearning.TrainAndEvaluateWithAnimalData using CPU ========");
            TransferLearning.TrainAndEvaluateWithAnimalData(device, true);

            Console.WriteLine("======== runing CifarResNet.Train using CPU ========");
            LSTMSequenceClassifier.Train(device, true);
        }
    }
}
