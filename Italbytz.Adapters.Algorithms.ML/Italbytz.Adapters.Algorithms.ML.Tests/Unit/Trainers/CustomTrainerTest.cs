using Microsoft.ML;

namespace Italbytz.ML.Tests.Unit.Trainers;

[TestClass]
public class CustomTrainerTest
{
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