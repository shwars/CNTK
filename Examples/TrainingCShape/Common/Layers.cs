using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    /// <summary>
    /// This is a proposed high level C# API. We welcome inputs from CNTK community. 
    /// </summary>
    public class Layers
    {
        static void Example()
        {
            // 1. prepare input variables
            var inputDimension = new int[] { 32, 32 };
            var outputDim = new int[] { 10 };
            Variable images = Variable.InputVariable(inputDimension, DataType.Float);
            Variable labels = Variable.InputVariable(outputDim, DataType.Float);

            // 2. prepare the layer with default hyper parameters
            var layer = new Layers(new Dictionary<string, object>
            {
                { "init", ParameterInitializationType.GlorotUniformInitializer },
                { "activation", Activation.Sigmoid },
                { "padding", true },
                { "strides", new int[] {2,2}}
            });

            // 3. compose models with CNTK high level APIs 
            // 3.1. build first convolution layer
            var conv1 = layer.Convolution(new int[] { 3, 3 })(images);
            var dropout1 = layer.Dropout()(conv1);

            // 3.2. build second convolution layer
            var conv2 = layer.Convolution(new int[] { 3, 3 })(dropout1);
            var dropout2 = layer.Dropout()(conv2);

            // 3.3. build the image classifier with a fully connected layer 
            Function imageClassifier = layer.Dense(outputDim)(dropout2);

            // 3.4. construct loss and prediction functions 
            Function lossFunction = CNTKLib.CrossEntropyWithSoftmax(imageClassifier, labels, "lossFunction");
            var prediction = CNTKLib.ClassificationError(imageClassifier, labels, "classificationError");

            var learningRatePerSample = new TrainingParameterScheduleDouble(0.01, TrainingParameterScheduleDouble.UnitType.Sample);
            IList<Learner> parameterLearners = new List<Learner>() { Learner.SGDLearner(imageClassifier.Parameters(), learningRatePerSample) };
            Trainer trainer = Trainer.CreateTrainer(imageClassifier, lossFunction, prediction, parameterLearners);

            // trainer.TrainMinibatch();
        }

        public Layers(DefaultOptions defaultOptions)
        {

        }

        public DefaultOptions Default
        {
            get;
            private set;
        }

        public enum Activation
        {
            Default,
            None,
            Sigmoid,
            Tanh
        }

        public enum ParameterInitializationType
        {
            Default,
            ConstantInitializer,
            GlorotUniformInitializer
        }

        public class DefaultOptions
        {
            public DefaultOptions(IEnumerable<KeyValuePair<string, object>> options)
            {
                throw new NotImplementedException();
            }

            public DefaultOptions(IDictionary<string, object> options)
            {
                throw new NotImplementedException();
            }

            public static implicit operator DefaultOptions(KeyValuePair<string, object>[] options)
            {
                return new DefaultOptions(options);
            }

            public static implicit operator DefaultOptions(Dictionary<string, object> options)
            {
                return new DefaultOptions(options);
            }
        }

        /// <summary>
        /// Build a Dense layer 
        /// </summary>
        /// <param name="shape">The shape of output variable</param>
        /// <param name="activation">activation type</param>
        /// <param name="init">initialization type</param>
        /// <param name="input_rank">dimensions starting to 0 to input_rank will be taken as input the dense layer </param>
        /// <param name="map_rank">dimensions starting from map_rank to shape rank will mapped to the output. 
        /// map_rank and input_rank cannot be both set </param>
        /// <param name="bias">a bias parameter will be include if true</param>
        /// <param name="init_bias">initial bias value</param>
        /// <param name="name">name of the node</param>
        /// <returns>a function that can be concatinated with another function or an input variable </returns>
        public Func<Variable, Function> Dense(NDShape shape, Activation activation= Activation.Default, ParameterInitializationType init = ParameterInitializationType.Default,
            int input_rank= 0, int map_rank= 0,
            bool bias= true, float init_bias= 0,
            string name= "")
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Build a Convolution layer 
        /// </summary>
        /// <param name="filterShape">shape of the convolution kernel</param>
        /// <param name="num_filters">number of filters</param>
        /// <param name="sequential">whether the input is to be sequential data</param>
        /// <param name="activation">activition type</param>
        /// <param name="init">parameter initialization type </param>
        /// <param name="pad">padding</param>
        /// <param name="strides">strides across the input space to apply the filter</param>
        /// <param name="bias">include bias parameter if true</param>
        /// <param name="init_bias">initial bias value</param>
        /// <param name="reduction_rank"></param>
        /// <param name="max_temp_mem_size_in_samples"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Func<Variable, Function> Convolution(NDShape filterShape,
                int numfilters= 0, 
                bool sequential= false, // time convolution if True (filter_shape[0] corresponds to dynamic axis)
                Activation activation = Activation.Default, ParameterInitializationType init = ParameterInitializationType.Default,
                bool pad= false,
                int strides= 1,
                bool bias= true,
                float init_bias= 0,
                int reduction_rank= 1, 
                int max_temp_mem_size_in_samples= 0,
                string name= "")
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Build a max pooling layer
        /// </summary>
        /// <param name="filter_shape">the filter shape</param>
        /// <param name="strides">strides to apply max pooling</param>
        /// <param name="pad">whether to do padding</param>
        /// <param name="name">name of the layer</param>
        /// <returns></returns>
        public Func<Variable, Function> MaxPooling(NDShape filter_shape,
           int strides = 1,
           bool pad = false,
           string name = "")
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Build an average pooling layer
        /// </summary>
        /// <param name="filter_shape">the filter shape</param>
        /// <param name="strides">strides to apply the filter</param>
        /// <param name="pad">whether to do padding</param>
        /// <param name="name">name of the layer</param>
        /// <returns></returns>
        public Function AveragePooling(NDShape filter_shape,
            int strides = 1,
            bool pad = false,
            string name = "")
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Build a dropout layer
        /// </summary>
        /// <param name="dropout_rate">the dropout rate</param>
        /// <param name="keep_prob">keep probability</param>
        /// <param name="name">name of the layer</param>
        /// <returns></returns>
        public Func<Variable, Function> Dropout(float dropout_rate= -1, float keep_prob= -1, string name= "")
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// Build an imbedding layer
        /// </summary>
        /// <param name="shape">shape of the output </param>
        /// <param name="init">parameter initialization type</param>
        /// <param name="weights">if provided, use the weight instead. The layer is not learnable. </param>
        /// <param name="name">name of the layer</param>
        /// <returns></returns>
        public Func<Variable, Function> Embedding(
            NDShape shape,
            ParameterInitializationType init = ParameterInitializationType.Default, IList<float> weights = null, string name= "")
        {
            throw new NotImplementedException();
        }

        public Func<Variable, Function> Recurrence(Function step_function, bool go_backwards= false, float initial_state= 0, 
            bool return_full_state= false, string name= "")
        {
            throw new NotImplementedException();
        }

        public Func<Variable, Function> RecurrenceFrom(Function step_function, bool go_backwards= false, 
            bool return_full_state= false, string name= "")
        {
            throw new NotImplementedException();
        }

        public Func<Variable, Function> Fold(Function folder_function, bool go_backwards= false, float initial_state= 0, 
            bool return_full_state= false, string name= "")
        {
            throw new NotImplementedException();
        }

        public Func<Variable, Function> UnfoldFrom(Function generator_function, Function ntil_predicate = null, int length_increase = 1, string name = "")
        {
            throw new NotImplementedException();
        }

        /// <summary>
        /// build a LSTM layer
        /// </summary>
        /// <param name="shape">output dimensions</param>
        /// <param name="cellShape">cell dimensions</param>
        /// <param name="activation">activition type</param>
        /// <param name="usePeepholes">use peepholes if true</param>
        /// <param name="init">parameter initialization types</param>
        /// <param name="initBias">initial bias value</param>
        /// <param name="enableSelfStabilization"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public Func<Variable, Function> LSTM(NDShape shape, NDShape cellShape= null, Activation activation = Activation.Tanh, 
            bool usePeepholes= false,
            ParameterInitializationType init = ParameterInitializationType.GlorotUniformInitializer, 
            float initBias= 0,
            bool enableSelfStabilization= false,
            string name= "")
        {
            throw new NotImplementedException();
        }

        public Func<Variable, Function> GRU(NDShape shape, NDShape cell_shape= null, Activation activation = Activation.Tanh,
            ParameterInitializationType init = ParameterInitializationType.GlorotUniformInitializer,
            float init_bias = 0,
            bool enable_self_stabilization= false,
            string name= "")
        {
            throw new NotImplementedException();
        }

        public Func<Variable, Function> RNNStep(NDShape shape, NDShape cell_shape= null, Activation activation = Activation.Sigmoid,
            ParameterInitializationType init = ParameterInitializationType.GlorotUniformInitializer,
            float init_bias = 0,
            bool enable_self_stabilization= false,
            string name= "")
        {
            throw new NotImplementedException();
        }

        public Func<Variable, Function> Delay(int T= 1, float initial_state= 0, string name= "")
        {
            throw new NotImplementedException();
        }

        public Func<Variable, Function> BatchNormalization(int map_rank= 0,
                   float init_scale=1,
                   int normalization_time_constant = 5000, 
                   int blend_time_constant=0,
                   double epsilon = 0.00001, 
                   bool use_cntk_engine = false,
                   string name="")
        {
            throw new NotImplementedException();
        }

        public Func<Variable, Function> LayerNormalization(float initial_scale = 1, float initial_bias = 0,
            double epsilon = 0.00001, string name = "")
        {
            throw new NotImplementedException();
        }

        public Func<Variable, Function> Stabilizer(int steepness=4, bool enable_self_stabilization=true, string name="")
        {
            throw new NotImplementedException();
        }
    }
}
