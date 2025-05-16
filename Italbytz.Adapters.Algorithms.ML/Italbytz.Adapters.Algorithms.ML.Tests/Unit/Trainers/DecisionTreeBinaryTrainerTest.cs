using Italbytz.ML.Tests.Data;
using Italbytz.ML.Trainers;
using JetBrains.Annotations;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Tests.Unit.Trainers;

[TestClass]
[TestSubject(typeof(DecisionTreeBinaryTrainer))]
public class DecisionTreeBinaryTrainerTest
{
    [TestMethod]
    public void TestRestaurantData()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        // data preparation
        var data = mlContext.Data.LoadFromTextFile<RestaurantModelInput>(
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                "Data/Restaurant", "restaurant_categories.csv"),
            ',', true);
        LookupMap<uint>[] lookupData =
        [
            new(0),
            new(1)
        ];
        var lookupIdvMap =
            mlContext.Data.LoadFromEnumerable(lookupData);
        // trainer and pipeline
        var trainer = new DecisionTreeBinaryTrainer();
        var pipeline = GetRestaurantPipeline(trainer, lookupIdvMap);
        var model = pipeline.Fit(data);
        var transformedData = model.Transform(data);
        // evaluation
        var metrics = mlContext.BinaryClassification
            .Evaluate(transformedData);
        Assert.AreEqual(1.0, metrics.Accuracy, 0.0001);
        Assert.AreEqual(1.0, metrics.AreaUnderRocCurve, 0.0001);
        Assert.AreEqual(1.0, metrics.F1Score, 0.0001);
        Assert.AreEqual(0.0, metrics.LogLoss, 0.0001);
        Assert.AreEqual(1.0, metrics.LogLossReduction, 0.0001);
        Assert.AreEqual(1.0, metrics.PositivePrecision, 0.0001);
        Assert.AreEqual(1.0, metrics.PositiveRecall, 0.0001);
        Assert.AreEqual(1.0, metrics.NegativePrecision, 0.0001);
        Assert.AreEqual(1.0, metrics.NegativeRecall, 0.0001);
    }


    private static EstimatorChain<ITransformer> GetRestaurantPipeline(
        IEstimator<ITransformer> trainer, IDataView lookupIdvMap)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"alternate", @"alternate"),
                new InputOutputColumnPair(@"bar", @"bar"),
                new InputOutputColumnPair(@"fri/sat", @"fri/sat"),
                new InputOutputColumnPair(@"hungry", @"hungry"),
                new InputOutputColumnPair(@"patrons", @"patrons"),
                new InputOutputColumnPair(@"price", @"price"),
                new InputOutputColumnPair(@"raining", @"raining"),
                new InputOutputColumnPair(@"reservation", @"reservation"),
                new InputOutputColumnPair(@"type", @"type"),
                new InputOutputColumnPair(@"wait_estimate", @"wait_estimate")
            })
            .Append(mlContext.Transforms.Concatenate(@"Features", @"alternate",
                @"bar", @"fri/sat", @"hungry", @"patrons", @"price", @"raining",
                @"reservation", @"type", @"wait_estimate"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Label",
                @"will_wait", keyData: lookupIdvMap))
            .Append(trainer);

        return pipeline;
    }
}