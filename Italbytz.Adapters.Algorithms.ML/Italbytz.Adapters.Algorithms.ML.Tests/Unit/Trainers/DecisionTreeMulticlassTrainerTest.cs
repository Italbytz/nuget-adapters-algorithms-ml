using Italbytz.ML.Tests.Data.CarEvaluation;
using Italbytz.ML.Tests.Data.NationalPoll;
using Italbytz.ML.Trainers;
using JetBrains.Annotations;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Tests.Unit.Trainers;

[TestClass]
[TestSubject(typeof(DecisionTreeBinaryTrainer))]
public class DecisionTreeMulticlassTrainerTest
{
    [TestMethod]
    public void TestNHPAData()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        // data preparation
        var data = mlContext.Data.LoadFromTextFile<NationalPollModelInput>(
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
                "Data/NationalPoll", "national_poll_on_healthy_aging_npha.csv"),
            ',', true);
        LookupMap<uint>[] lookupData =
        [
            new(1),
            new(2),
            new(3)
        ];
        var lookupIdvMap =
            mlContext.Data.LoadFromEnumerable(lookupData);
        // trainer and pipeline
        var trainer =
            new DecisionTreeMulticlassTrainer<TernaryClassificationOutput>();
        var pipeline = GetNHPAPipeline(trainer, lookupIdvMap);
        var model = pipeline.Fit(data);
        var transformedData = model.Transform(data);
        // evaluation
        var metrics = mlContext.MulticlassClassification
            .Evaluate(transformedData);
        Assert.AreEqual(0.93706, metrics.MacroAccuracy, 0.0001);
        Assert.AreEqual(0.94537, metrics.MicroAccuracy, 0.0001);
        Assert.AreEqual(1.88657, metrics.LogLoss, 0.0001);
        Assert.AreEqual(-0.86595, metrics.LogLossReduction, 0.0001);
    }


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
        var trainer =
            new DecisionTreeMulticlassTrainer<QuaternaryClassificationOutput>();
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

    protected EstimatorChain<ITransformer?> GetNHPAPipeline(
        IEstimator<ITransformer> trainer, IDataView lookupIdvMap)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var pipeline = mlContext.Transforms.ReplaceMissingValues(new[]
            {
                new InputOutputColumnPair(@"Age", @"Age"),
                new InputOutputColumnPair(@"Physical_Health",
                    @"Physical_Health"),
                new InputOutputColumnPair(@"Mental_Health", @"Mental_Health"),
                new InputOutputColumnPair(@"Dental_Health", @"Dental_Health"),
                new InputOutputColumnPair(@"Employment", @"Employment"),
                new InputOutputColumnPair(@"Stress_Keeps_Patient_from_Sleeping",
                    @"Stress_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(
                    @"Medication_Keeps_Patient_from_Sleeping",
                    @"Medication_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(@"Pain_Keeps_Patient_from_Sleeping",
                    @"Pain_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(
                    @"Bathroom_Needs_Keeps_Patient_from_Sleeping",
                    @"Bathroom_Needs_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(@"Uknown_Keeps_Patient_from_Sleeping",
                    @"Uknown_Keeps_Patient_from_Sleeping"),
                new InputOutputColumnPair(@"Trouble_Sleeping",
                    @"Trouble_Sleeping"),
                new InputOutputColumnPair(@"Prescription_Sleep_Medication",
                    @"Prescription_Sleep_Medication"),
                new InputOutputColumnPair(@"Race", @"Race"),
                new InputOutputColumnPair(@"Gender", @"Gender")
            })
            .Append(mlContext.Transforms.Concatenate(@"Features", @"Age",
                @"Physical_Health", @"Mental_Health", @"Dental_Health",
                @"Employment", @"Stress_Keeps_Patient_from_Sleeping",
                @"Medication_Keeps_Patient_from_Sleeping",
                @"Pain_Keeps_Patient_from_Sleeping",
                @"Bathroom_Needs_Keeps_Patient_from_Sleeping",
                @"Uknown_Keeps_Patient_from_Sleeping", @"Trouble_Sleeping",
                @"Prescription_Sleep_Medication", @"Race", @"Gender"))
            .Append(mlContext.Transforms.Conversion.MapValueToKey(
                @"Label", @"Number_of_Doctors_Visited",
                keyData: lookupIdvMap))
            .Append(trainer);

        return pipeline;
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