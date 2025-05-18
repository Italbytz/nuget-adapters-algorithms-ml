using Italbytz.ML.Tests.Data.LSExample;
using Italbytz.ML.Tests.Data.TaxiFare;
using Italbytz.ML.Trainers;
using JetBrains.Annotations;
using Microsoft.ML;
using Microsoft.ML.Data;
using InputOutputColumnPair = Microsoft.ML.InputOutputColumnPair;
using ITransformer = Microsoft.ML.ITransformer;

namespace Italbytz.ML.Tests.Unit.Trainers;

[TestClass]
[TestSubject(typeof(LeastSquaresTrainer))]
public class LeastSquaresTrainerTest
{
    [TestMethod]
    public void TestLSExampleData()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        // data preparation
        var data = mlContext.Data.LoadFromTextFile<LSExampleModelInput>(
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                "Data/LSExample", "LSExample.csv"),
            ',', true);
        var trainer =
            new LeastSquaresTrainer();
        var pipeline = GetLSExamplePipeline(trainer);
        var model = pipeline.Fit(data);
        var transformedData = model.Transform(data);
        // evaluation
        var metrics = mlContext.Regression
            .Evaluate(transformedData);
        Assert.AreEqual(0.0, metrics.MeanAbsoluteError, 0.0001);
        Assert.AreEqual(0.0, metrics.MeanSquaredError, 0.0001);
        Assert.AreEqual(0.0, metrics.RootMeanSquaredError, 0.0001);
        Assert.AreEqual(0.0, metrics.LossFunction, 0.0001);
        Assert.AreEqual(1.0, metrics.RSquared, 0.0001);
    }

    public void TestTaxiData()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        // data preparation
        var data = mlContext.Data.LoadFromTextFile<TaxiModelInput>(Path.Combine(
                AppDomain.CurrentDomain.BaseDirectory,
                "Data/TaxiFare", "taxi-fare-train.csv"),
            ',', true);
        var trainer =
            new LeastSquaresTrainer();
        var pipeline = GetTaxiFarePipeline(trainer);
        var model = pipeline.Fit(data);
        var transformedData = model.Transform(data);
        // evaluation
        var metrics = mlContext.Regression
            .Evaluate(transformedData);
        Assert.AreEqual(0.0001, metrics.MeanAbsoluteError, 0.0001);
    }

    private EstimatorChain<ITransformer?> GetLSExamplePipeline(
        LeastSquaresTrainer trainer)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"x1", @"x1"),
                new InputOutputColumnPair(@"x2", @"x2"),
                new InputOutputColumnPair(@"Label", @"y")
            })
            .Append(mlContext.Transforms.Concatenate(@"Features", @"x1", @"x2"))
            .Append(trainer);
        return pipeline;
    }

    private EstimatorChain<ITransformer?> GetTaxiFarePipeline(
        LeastSquaresTrainer trainer)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var pipeline = mlContext.Transforms.Categorical.OneHotEncoding(
                new[]
                {
                    new InputOutputColumnPair(@"vendor_id", @"vendor_id"),
                    new InputOutputColumnPair(@"payment_type", @"payment_type"),
                    new InputOutputColumnPair(@"Label", @"fare_amount")
                })
            .Append(mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"rate_code", @"rate_code"),
                new InputOutputColumnPair(@"passenger_count",
                    @"passenger_count"),
                new InputOutputColumnPair(@"trip_time_in_secs",
                    @"trip_time_in_secs"),
                new InputOutputColumnPair(@"trip_distance", @"trip_distance"),
                new InputOutputColumnPair(@"Label", @"fare_amount")
            }))
            .Append(mlContext.Transforms.Concatenate(@"Features", @"vendor_id",
                @"payment_type", @"rate_code", @"passenger_count",
                @"trip_time_in_secs", @"trip_distance")).Append(trainer);

        return pipeline;
    }
}