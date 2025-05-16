using Italbytz.ML.Tests.Data.TaxiFare;
using Italbytz.ML.Trainers;
using JetBrains.Annotations;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Tests.Unit.Trainers;

[TestClass]
[TestSubject(typeof(LeastSquaresTrainer))]
public class LeastSquaresTrainerTest
{
    [TestMethod]
    public void TestTaxiData()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        // data preparation
        var data = mlContext.Data.LoadFromTextFile<TaxiModelInput>(
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                "Data/TaxiFare", "taxi-fare-train.csv"),
            ',', true);
        var trainer =
            new DecisionTreeMulticlassTrainer<TernaryClassificationOutput>();
        var pipeline = GetTaxiFarePipeline(trainer);
        var model = pipeline.Fit(data);
        var transformedData = model.Transform(data);
        // evaluation
        var metrics = mlContext.Regression
            .Evaluate(transformedData);
        Assert.AreEqual(0.0001, metrics.MeanAbsoluteError, 0.0001);
    }

    private EstimatorChain<ITransformer?> GetTaxiFarePipeline(
        DecisionTreeMulticlassTrainer<TernaryClassificationOutput> trainer)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                new[]
                {
                    new InputOutputColumnPair(@"vendor_id", @"vendor_id"),
                    new InputOutputColumnPair(@"payment_type", @"payment_type")
                })
            .Append(mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"rate_code", @"rate_code"),
                new InputOutputColumnPair(@"passenger_count",
                    @"passenger_count"),
                new InputOutputColumnPair(@"trip_time_in_secs",
                    @"trip_time_in_secs"),
                new InputOutputColumnPair(@"trip_distance", @"trip_distance")
            }))
            .Append(mlContext.Transforms.Concatenate(@"Features", @"vendor_id",
                @"payment_type", @"rate_code", @"passenger_count",
                @"trip_time_in_secs", @"trip_distance")).Append(trainer);

        return pipeline;
    }
}