using Italbytz.ML.Tests.Data;
using Italbytz.ML.Trainers;
using JetBrains.Annotations;
using logicGP.Tests.Data.Real;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Tests.Unit.Trainers;

[TestClass]
[TestSubject(typeof(DecisionTreeBinaryTrainer))]
public class DecisionTreeBinaryTrainerTest
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

    [TestMethod]
    public void TestCustomMapper()
    {
        var mlContext = new MLContext();

        var samples = new List<InputData>
        {
            new() { Age = 16 },
            new() { Age = 35 },
            new() { Age = 60 },
            new() { Age = 28 }
        };

        var data = mlContext.Data.LoadFromEnumerable(samples);

        void Mapping(InputData input, CustomMappingOutput output)
        {
            output.AgeName = input.Age switch
            {
                < 18 => "Child",
                < 55 => "Man",
                _ => "Grandpa"
            };
        }

        var pipeline =
            mlContext.Transforms.CustomMapping(
                (Action<InputData, CustomMappingOutput>)Mapping, null);

        var transformer = pipeline.Fit(data);
        var transformedData = transformer.Transform(data);

        var dataEnumerable =
            mlContext.Data.CreateEnumerable<TransformedData>(transformedData,
                false);

        var dataArray = dataEnumerable.ToArray();

        Assert.AreEqual(4, dataArray.Length);
        Assert.AreEqual("Child", dataArray[0].AgeName);
        Assert.AreEqual("Man", dataArray[1].AgeName);
        Assert.AreEqual("Grandpa", dataArray[2].AgeName);
        Assert.AreEqual("Man", dataArray[3].AgeName);
    }
}

internal class InputData
{
    public int Age { get; set; }
}

internal class CustomMappingOutput
{
    public string? AgeName { get; set; }
}

internal class TransformedData
{
    public int Age { get; set; }

    public string? AgeName { get; set; }
}