// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers.LightGbm;

namespace Microsoft.ML.AutoML
{
    using ITrainerEstimator = ITrainerEstimator<ISingleFeaturePredictionTransformer<object>, object>;

    internal class RandomizedPcaExtension : ITrainerExtension
    {
        private const int DefaultRank = 1;

        public IEnumerable<SweepableParam> GetHyperparamSweepRanges()
        {
            return SweepableParams.BuildPcaParams();
        }

        public ITrainerEstimator CreateInstance(MLContext mlContext, IEnumerable<SweepableParam> sweepParams,
            ColumnInformation columnInfo)
        {
            RandomizedPcaTrainer.Options options = null;
            if (sweepParams == null || !sweepParams.Any())
            {
                options = new RandomizedPcaTrainer.Options();
                options.Rank = DefaultRank;
            } else
            {
                options = TrainerExtensionUtil.CreateOptions<RandomizedPcaTrainer.Options>(sweepParams);
            }
            return mlContext.AnomalyDetection.Trainers.RandomizedPca(options);
        }

        public PipelineNode CreatePipelineNode(IEnumerable<SweepableParam> sweepParams, ColumnInformation columnInfo)
        {
            return TrainerExtensionUtil.BuildPipelineNode(TrainerExtensionCatalog.GetTrainerName(this), sweepParams,
                columnInfo.LabelColumnName, columnInfo.ExampleWeightColumnName);
        }
    }
}