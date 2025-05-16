using Italbytz.ML.Trainers;
using JetBrains.Annotations;
using logicGP.Tests.Data.Real;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Tests.Unit.Trainers;

[TestClass]
[TestSubject(typeof(DecisionTreeBinaryTrainer))]
public class DecisionTreeMulticlassTrainerTest
{
    [TestMethod]
    public void TestCarEvaluationData()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        // data preparation
        var data = mlContext.Data.LoadFromTextFile<CarEvaluationModelInput>(
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                "Data/CarEvaluation", "car_evaluation_strings.csv"),
            ',', true);
        var lookupData = new[]
        {
            new LookupMap<string>("unacc"),
            new LookupMap<string>("acc"),
            new LookupMap<string>("good"),
            new LookupMap<string>("vgood")
        };
        var lookupIdvMap =
            mlContext.Data.LoadFromEnumerable(lookupData);
        // trainer and pipeline
        var trainer = new DecisionTreeBinaryTrainer();
        var pipeline = GetCarEvaluationPipeline(trainer, lookupIdvMap);
        var model = pipeline.Fit(data);
        var transformedData = model.Transform(data);
        // evaluation
        var metrics = mlContext.MulticlassClassification
            .Evaluate(transformedData);
        Assert.AreEqual(1.0, metrics.MacroAccuracy, 0.0001);
        Assert.AreEqual(1.0, metrics.MicroAccuracy, 0.0001);
        Assert.AreEqual(0.0, metrics.LogLoss, 0.0001);
        Assert.AreEqual(1.0, metrics.LogLossReduction, 0.0001);
    }

    protected EstimatorChain<ITransformer?> GetCarEvaluationPipeline(
        IEstimator<ITransformer> trainer, IDataView lookupIdvMap)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;

        var buyingLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "low" },
            new CategoryLookupMap { Value = 1f, Category = "med" },
            new CategoryLookupMap { Value = 2f, Category = "high" },
            new CategoryLookupMap { Value = 3f, Category = "vhigh" }
        };
        var buyingLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(buyingLookupData);

        var maintLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "low" },
            new CategoryLookupMap { Value = 1f, Category = "med" },
            new CategoryLookupMap { Value = 2f, Category = "high" },
            new CategoryLookupMap { Value = 3f, Category = "vhigh" }
        };
        var maintLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(maintLookupData);

        var doorsLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "two" },
            new CategoryLookupMap { Value = 1f, Category = "three" },
            new CategoryLookupMap { Value = 2f, Category = "four" },
            new CategoryLookupMap { Value = 3f, Category = "fiveormore" }
        };
        var doorsLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(doorsLookupData);

        var personsLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "two" },
            new CategoryLookupMap { Value = 1f, Category = "four" },
            new CategoryLookupMap { Value = 2f, Category = "more" }
        };
        var personsLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(personsLookupData);

        var lugBootLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "small" },
            new CategoryLookupMap { Value = 1f, Category = "med" },
            new CategoryLookupMap { Value = 2f, Category = "big" }
        };
        var lugBootLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(lugBootLookupData);

        var safetyLookupData = new[]
        {
            new CategoryLookupMap { Value = 0f, Category = "low" },
            new CategoryLookupMap { Value = 1f, Category = "med" },
            new CategoryLookupMap { Value = 2f, Category = "high" }
        };
        var safetyLookupIdvMap =
            mlContext.Data.LoadFromEnumerable(safetyLookupData);

        var pipeline =
            mlContext.Transforms.Conversion.MapValue("buying",
                    buyingLookupIdvMap, buyingLookupIdvMap.Schema["Category"],
                    buyingLookupIdvMap.Schema[
                        "Value"], "buying")
                .Append(mlContext.Transforms.Conversion.MapValue("maint",
                    maintLookupIdvMap, maintLookupIdvMap.Schema["Category"],
                    maintLookupIdvMap.Schema[
                        "Value"], "maint"))
                .Append(mlContext.Transforms.Conversion.MapValue("doors",
                    doorsLookupIdvMap, doorsLookupIdvMap.Schema["Category"],
                    doorsLookupIdvMap.Schema[
                        "Value"], "doors"))
                .Append(mlContext.Transforms.Conversion.MapValue("persons",
                    personsLookupIdvMap, personsLookupIdvMap.Schema["Category"],
                    personsLookupIdvMap.Schema[
                        "Value"], "persons"))
                .Append(mlContext.Transforms.Conversion.MapValue("lug_boot",
                    lugBootLookupIdvMap, lugBootLookupIdvMap.Schema["Category"],
                    lugBootLookupIdvMap.Schema[
                        "Value"], "lug_boot"))
                .Append(mlContext.Transforms.Conversion.MapValue("safety",
                    safetyLookupIdvMap, safetyLookupIdvMap.Schema["Category"],
                    safetyLookupIdvMap.Schema[
                        "Value"], "safety"))
                .Append(mlContext.Transforms.Concatenate(@"Features", @"buying",
                    @"maint", @"doors", @"persons", @"lug_boot", @"safety"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Label",
                    @"class", keyData: lookupIdvMap))
                .Append(trainer);

        return pipeline;
    }
}