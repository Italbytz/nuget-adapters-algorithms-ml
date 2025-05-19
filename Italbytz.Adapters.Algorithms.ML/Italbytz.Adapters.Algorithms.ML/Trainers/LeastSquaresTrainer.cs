using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Italbytz.ML.Trainers;

public class
    LeastSquaresTrainer : CustomRegressionTrainer
{
    private double[]? _parameters;

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

    protected override
        CustomMappingEstimator<RegressionInput, RegressionOutput>
        GetCustomMappingEstimator()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var mapping = new LeastSquaresMapping(_parameters);
        return mlContext.Transforms
            .CustomMapping(
                mapping
                    .GetMapping<RegressionInput,
                        RegressionOutput>(), null);
    }
}