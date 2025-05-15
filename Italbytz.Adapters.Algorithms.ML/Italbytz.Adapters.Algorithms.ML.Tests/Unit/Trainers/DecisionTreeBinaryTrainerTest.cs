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
    private readonly IDataView _data;

    private readonly LookupMap<uint>[] _lookupData =
    [
        new(0),
        new(1)
    ];

    public DecisionTreeBinaryTrainerTest()
    {
        var mlContext = new MLContext();
        var path = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "Data/Restaurant", "restaurant_categories.csv");
        _data = mlContext.Data.LoadFromTextFile<RestaurantModelInput>(
            path,
            ',', true);
    }

    [TestMethod]
    public void TestInducedTreeClassifiesDataSetCorrectly()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var lookupIdvMap = mlContext.Data.LoadFromEnumerable(_lookupData);
        var trainer = new DecisionTreeBinaryTrainer();
        var pipeline = GetPipeline(trainer, lookupIdvMap);
        var model = pipeline.Fit(_data);
        var transformedData = model.Transform(_data);
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

    protected EstimatorChain<ITransformer> GetPipeline(
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
    public string AgeName { get; set; }
}

internal class TransformedData
{
    public int Age { get; set; }

    public string AgeName { get; set; }
}