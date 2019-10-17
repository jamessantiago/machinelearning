using System;
using System.Collections.Generic;
using System.Text;
using Microsoft.ML.Data;

namespace Microsoft.ML.AutoML.Experiment.MetricsAgents
{
    internal class AnomalyMetricsAgent : IMetricsAgent<FakeAnomalyDetectionMetrics>
    {
        private readonly MLContext _mlContext;
        private readonly AnomalyDetectionMetric _optimizingMetric;

        public AnomalyMetricsAgent(MLContext mlContext,
            AnomalyDetectionMetric optimizingMetric)
        {
            _mlContext = mlContext;
            _optimizingMetric = optimizingMetric;
        }

        public double GetScore(FakeAnomalyDetectionMetrics metrics)
        {
            if (metrics == null && _optimizingMetric != AnomalyDetectionMetric.FakeAccuracy)
            {
                return double.NaN;
            }

            switch (_optimizingMetric)
            {
                case AnomalyDetectionMetric.FakeAccuracy:
                    return 1;
                case AnomalyDetectionMetric.AreaUnderRocCurve:
                    return metrics.AreaUnderRocCurve;
                case AnomalyDetectionMetric.DetectionRateAtFalsePositiveCount:
                    return metrics.DetectionRateAtFalsePositiveCount;
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
                case AnomalyDetectionMetric.FakeAccuracy:
                    return score == 1;
                case AnomalyDetectionMetric.AreaUnderRocCurve:
                    return score == 1;
                case AnomalyDetectionMetric.DetectionRateAtFalsePositiveCount:
                    return score == 1;
                default:
                    throw MetricsAgentUtil.BuildMetricNotSupportedException(_optimizingMetric);
            }
        }

        public FakeAnomalyDetectionMetrics EvaluateMetrics(IDataView data, string labelColumn)
        {
            if (_optimizingMetric == AnomalyDetectionMetric.FakeAccuracy)
                return new FakeAnomalyDetectionMetrics { FakeAccuracy = 1 };

            var transform = _mlContext.Transforms.Conversion.ConvertType(inputColumnName: "PredictedLabel", outputColumnName: "Label", outputKind: DataKind.Single);
            var transformedData = transform.Fit(data).Transform(data);
            try
            {
                var realMetrics = _mlContext.AnomalyDetection.Evaluate(transformedData);
                var metrics = new FakeAnomalyDetectionMetrics
                {
                    FakeAccuracy = 1,
                    AreaUnderRocCurve = realMetrics.AreaUnderRocCurve,
                    DetectionRateAtFalsePositiveCount = realMetrics.DetectionRateAtFalsePositiveCount
                };
                return metrics;
            } catch
            {
                return null;
            }
        }
    }
}
