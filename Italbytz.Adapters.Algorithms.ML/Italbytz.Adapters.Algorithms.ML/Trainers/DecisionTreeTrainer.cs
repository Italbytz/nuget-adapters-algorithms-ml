using System;
using System.Collections.Generic;
using System.Globalization;
using Italbytz.AI.Learning;
using Italbytz.AI.Learning.Framework;
using Italbytz.AI.Learning.Learners;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Trainers;

public abstract class
    DecisionTreeTrainer<TOutput> : CustomClassificationTrainer<TOutput>
    where TOutput : class, new()
{
    protected readonly ILearner Learner;
    protected IDataExcerpt? DataExcerpt;
    protected IDataSetSpecification? Spec;

    protected DecisionTreeTrainer()
    {
        Learner = new DecisionTreeLearner();
    }

    /// <inheritdoc />
    protected override void PrepareForFit(IDataView input)
    {
        DataExcerpt = input.GetDataExcerpt();
        Spec = DataExcerpt.GetDataSetSpecification();
        var dataSet = DataExcerpt.GetDataSet(Spec);
        Learner.Train(dataSet);
    }

    protected override void Map(ClassificationInput input, TOutput output)
    {
        var example = ToExample(input);
        var prediction = Learner.Predict(example);
        switch (output)
        {
            case IBinaryClassificationOutput binaryOutput:
                binaryOutput.PredictedLabel =
                    uint.Parse(prediction, CultureInfo.InvariantCulture);
                binaryOutput.Score =
                    prediction.Equals("1", StringComparison.OrdinalIgnoreCase)
                        ? 0f
                        : 1f;
                binaryOutput.Probability =
                    prediction.Equals("1", StringComparison.OrdinalIgnoreCase)
                        ? 0f
                        : 1f;
                break;
            case IMulticlassClassificationOutput multiclassOutput:
            {
                var classes = DataExcerpt.UniqueLabelValues.Length;
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
        var featureNames = DataExcerpt?.FeatureNames;
        var index = 0;
        foreach (var featureName in featureNames!)
        {
            attributes.Add(featureName,
                new StringAttribute(
                    features[index].ToString(CultureInfo.InvariantCulture),
                    Spec!.GetAttributeSpecFor(featureName)));
            index++;
        }

        // Mock target attribute
        attributes.Add(DefaultColumnNames.Label, new StringAttribute("1",
            Spec!.GetAttributeSpecFor(DefaultColumnNames.Label)));
        return new Example(attributes,
            attributes[DefaultColumnNames.Label]);
    }
}