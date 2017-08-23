using System.Collections.Generic;

namespace CNTK
{
    public partial class MinibatchSourceConfig
    {
        /// <summary>
        /// create a MinibatchSourceConfig with a collection of deserializer transforms
        /// </summary>
        /// <param name="deserializers">deserializer transforms</param>
        public MinibatchSourceConfig(IList<CNTKDictionary> deserializers) : this(Helper.AsDictionaryVector(deserializers))
        {
        }
    }
}
