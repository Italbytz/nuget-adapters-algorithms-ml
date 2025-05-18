using System;
using System.Collections.Generic;
using System.Globalization;
using Italbytz.AI.Learning;
using Italbytz.AI.Learning.Framework;
using Microsoft.ML.Data;

namespace Italbytz.ML.Trainers;

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
        switch (arg2)
        {
            case IBinaryClassificationOutput output:
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
                break;
            case IMulticlassClassificationOutput multiclassOutput:
            {
                var classes = dataExcerpt.UniqueLabelValues.Length;
                var predictedLabel = uint.Parse(prediction,
                    CultureInfo.InvariantCulture);
                multiclassOutput.PredictedLabel = predictedLabel;
                var scores = new float[classes];
                scores[predictedLabel - 1] = 1f;
                var probabilities = new float[classes];
                probabilities[predictedLabel - 1] = 1f;
                multiclassOutput.Score =
                    new VBuffer<float>(scores.Length, scores);
                multiclassOutput.Probability =
                    new VBuffer<float>(probabilities.Length, probabilities);
                break;
            }
            default:
                throw new ArgumentException(
                    "The destination is not of type IBinaryClassificationOutput or IMulticlassClassificationOutput");
        }
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