using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Experiment.MetricsAgents
{
    internal class AnomalyMetricsAgent : IMetricsAgent<AnomalyDetectionMetrics>
    {
        private readonly MLContext _mlContext;
        private readonly AnomalyDetectionMetric _optimizingMetric;

        public AnomalyMetricsAgent(MLContext mlContext,
            AnomalyDetectionMetric optimizingMetric)
        {
            _mlContext = mlContext;
            _optimizingMetric = optimizingMetric;
        }

        public double GetScore(AnomalyDetectionMetrics metrics)
        {
            if (metrics == null)
            {
                return double.NaN;
            }

            switch (_optimizingMetric)
            {
                case AnomalyDetectionMetric.AreaUnderRocCurve:
                    return metrics.AreaUnderRocCurve;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public bool IsModelPerfect(double score)
        {
            if (double.IsNaN(score))
            {
                return false;
            }

            switch (_optimizingMetric)
            {
                case AnomalyDetectionMetric.AreaUnderRocCurve:
                    return score == 1;
                case AnomalyDetectionMetric.DetectionRateAtFalsePositiveCount:
                    return score == 1;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public AnomalyDetectionMetrics EvaluateMetrics(IDataView data, string labelColumn)
        {
            //var keyList = new List<float>() { 1.0F, 2.0F };
            //var valueList = new List<float>() { 0.0F, 1.0F };
            //var converter = _mlContext.Transforms.Conversion.MapValue("LabelValue", new [] { new KeyValuePair<float,float>(1.0F, 0.0F) }, inputColumnName: "Label");
            //converter.Append(_mlContext.Transforms.Conversion.MapValue("LabelValue", new[] { new KeyValuePair<float, float>(2.0F, 1.0F) }, inputColumnName: "Label"));
            //var results = converter.Fit(data).Transform(data);
            //var preview = results.Preview();

            var transform = _mlContext.Transforms.Conversion.ConvertType(inputColumnName: "PredictedLabel", outputColumnName: "Label", outputKind: DataKind.Single);
            var transformedData = transform.Fit(data).Transform(data);
            try
            {
                return _mlContext.AnomalyDetection.Evaluate(transformedData);
            } catch
            {
                return null;
            }
        }
    }
}
