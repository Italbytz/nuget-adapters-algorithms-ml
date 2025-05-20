using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML.Trainers;

public class
    LeastSquaresTrainer : CustomRegressionTrainer
{
    private double[]? _parameters;

    /// <inheritdoc />
    protected override void PrepareForFit(IDataView input)
    {
        /*var dataExcerpt = input.GetDataExcerpt();
        var parameters = MathNet.Numerics.Fit.MultiDim(
            dataExcerpt.Features.ToArray()
                .Select(e => e.Select(f => (double)f).ToArray()).ToArray(),
            dataExcerpt.Labels.ToArray().Select(e => (double)e).ToArray(),
            true);*/
        var features = input
            .GetColumn<float[]>(DefaultColumnNames.Features)
            .ToList();
        var labels = input
            .GetColumn<float>(DefaultColumnNames.Label)
            .ToList();
        var x = features.Select(feature =>
            feature.Select(entry => (double)entry).ToArray()).ToArray();
        var y = labels.Select(label => (double)label).ToArray();
        _parameters = MathNet.Numerics.Fit.MultiDim(x
            , y, true);
    }

    /// <inheritdoc />
    protected override void Map(RegressionInput input, RegressionOutput output)
    {
        var score = _parameters![0];
        var features = input.Features;
        for (var i = 1; i < _parameters.Length; i++)
            score += _parameters[i] * features[i - 1];
        output.Score = (float)score;
    }
}