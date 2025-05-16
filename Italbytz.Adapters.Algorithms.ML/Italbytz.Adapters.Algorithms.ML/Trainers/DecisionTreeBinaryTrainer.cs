using System;
using System.Collections.Generic;
using System.Globalization;
using Italbytz.AI.Learning;
using Italbytz.AI.Learning.Framework;
using Microsoft.ML;

namespace Italbytz.ML.Trainers;

/// <inheritdoc />
public class DecisionTreeBinaryTrainer : DecisionTreeTrainer<
    BinaryClassificationInput,
    BinaryClassificationOutput>
{
    /// <inheritdoc />
    public override SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var mapping = new DecisionTreeMapping(_learner, null, null);
        return mlContext.Transforms
            .CustomMapping(
                mapping
                    .GetMapping<BinaryClassificationInput,
                        BinaryClassificationOutput>(), null)
            .GetOutputSchema(inputSchema);
    }

    /// <inheritdoc />
    public override ITransformer Fit(IDataView input)
    {
        var dataExcerpt = input.GetDataExcerpt();
        var spec =
            dataExcerpt.GetDataSetSpecification();
        var dataSet = dataExcerpt.GetDataSet(spec);
        _learner.Train(dataSet);
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var mapping = new DecisionTreeMapping(_learner, dataExcerpt, spec);
        return mlContext.Transforms
            .CustomMapping(
                mapping
                    .GetMapping<BinaryClassificationInput,
                        BinaryClassificationOutput>(), null).Fit(input);
    }
}

public class DecisionTreeMapping(
    ILearner learner,
    IDataExcerpt? dataExcerpt,
    IDataSetSpecification? spec)
{
    public Action<TSrc, TDst> GetMapping<TSrc, TDst>()
        where TSrc : class, new() where TDst : class, new()
    {
        return Map<TSrc, TDst>;
    }

    private void Map<TSrc, TDst>(TSrc arg1, TDst arg2)
        where TSrc : class, new() where TDst : class, new()
    {
        var example = ToExample(arg1);
        var prediction = learner.Predict(example);
        if (arg2 is not IBinaryClassificationOutput output)
            return;
        output.PredictedLabel =
            uint.Parse(prediction, CultureInfo.InvariantCulture);
        output.Score =
            prediction.Equals("1", StringComparison.OrdinalIgnoreCase)
                ? 0f
                : 1f;
        output.Probability =
            prediction.Equals("1", StringComparison.OrdinalIgnoreCase)
                ? 0f
                : 1f;
    }

    private IExample ToExample<TSrc>(TSrc src) where TSrc : class, new()
    {
        if (src is not ICustomMappingInput input)
            throw new ArgumentException(
                "The input is not of type ICustomMappingInputSchema");
        var features = input.Features;
        Dictionary<string, IAttribute> attributes = new();
        var featureNames = dataExcerpt?.FeatureNames;
        var index = 0;
        foreach (var featureName in featureNames!)
        {
            attributes.Add(featureName,
                new StringAttribute(
                    features[index].ToString(CultureInfo.InvariantCulture),
                    spec!.GetAttributeSpecFor(featureName)));
            index++;
        }

        // Mock target attribute
        attributes.Add(DefaultColumnNames.Label, new StringAttribute("1",
            spec!.GetAttributeSpecFor(DefaultColumnNames.Label)));
        return new Example(attributes,
            attributes[DefaultColumnNames.Label]);
    }
}