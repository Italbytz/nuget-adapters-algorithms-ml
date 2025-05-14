using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Italbytz.AI.Learning;
using Italbytz.AI.Learning.Framework;

namespace Italbytz.ML;

/// <summary>
///     Extensions for <see cref="IDataExcerpt" />.
/// </summary>
public static class IDataExcerptExtensions
{
    /// <summary>
    ///     Gets the data set specification for the <see cref="IDataExcerpt" />.
    /// </summary>
    public static IDataSetSpecification GetDataSetSpecification(
        this IDataExcerpt dataExcerpt)
    {
        var featureNames = dataExcerpt.FeatureNames;
        var dss = new DataSetSpecification();
        foreach (var featureName in featureNames)
            dss.DefineStringAttribute(featureName,
                dataExcerpt.GetUniqueFeatureValues(featureName)
                    .Select(v => v.ToString(CultureInfo.InvariantCulture))
                    .ToArray());
        dss.DefineStringAttribute(DefaultColumnNames.Label,
            dataExcerpt.UniqueLabelValues
                .Select(v => v.ToString(CultureInfo.InvariantCulture))
                .ToArray());
        return dss;
    }

    /// <summary>
    ///     Converts an <see cref="IDataExcerpt" /> to a <see cref="DataSet" />.
    /// </summary>
    /// <param name="spec">The data set specification.</param>
    /// <returns>
    ///     A <see cref="DataSet" /> representing the data in the
    ///     <see cref="IDataExcerpt" />.
    /// </returns>
    /// <remarks>
    ///     This method creates a new <see cref="DataSet" /> and populates it with
    ///     the data from the <see cref="IDataExcerpt" />.
    /// </remarks>
    public static IDataSet GetDataSet(this IDataExcerpt dataExcerpt,
        IDataSetSpecification spec)
    {
        var features = dataExcerpt.Features;
        var featureNames = dataExcerpt.FeatureNames;
        var labels = dataExcerpt.Labels;
        var dataSet = new DataSet(spec);
        // Iterate through rows
        var rowIndex = 0;
        foreach (var feature in features)
        {
            Dictionary<string, IAttribute> attributes = new();
            // Iterate through columns
            var columnIndex = 0;
            foreach (var featureName in featureNames)
            {
                var columnSpecification =
                    spec.GetAttributeSpecFor(featureName);
                var value = feature[columnIndex];
                attributes.Add(featureName,
                    new StringAttribute(
                        value.ToString(CultureInfo.InvariantCulture),
                        columnSpecification));
                columnIndex++;
            }

            var targetAttribute = new StringAttribute(
                labels[rowIndex].ToString(),
                spec.GetAttributeSpecFor(DefaultColumnNames.Label));
            attributes.Add(DefaultColumnNames.Label, targetAttribute);
            var example = new Example(attributes, targetAttribute);
            dataSet.Examples.Add(example);
            rowIndex++;
        }

        return dataSet;
    }
}