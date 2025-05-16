using System;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace Italbytz.ML.Trainers;

// ToDo: Replace BinaryClassificationInput with the actual input type for your model
public class
    LeastSquaresTrainer : CustomTrainer<BinaryClassificationInput,
    RegressionOutput>
{
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
        /*var parameters = MathNet.Numerics.Fit.MultiDim(features.ToArray(),
            labels.ToArray(), true);*/
    }

    protected override
        CustomMappingEstimator<BinaryClassificationInput, RegressionOutput>
        GetCustomMappingEstimator()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var mapping = new LeastSquaresMapping();
        return mlContext.Transforms
            .CustomMapping(
                mapping
                    .GetMapping<BinaryClassificationInput,
                        RegressionOutput>(), null);
    }
}

public class LeastSquaresMapping
{
    public Action<TSrc, TDst> GetMapping<TSrc, TDst>()
        where TSrc : class, new() where TDst : class, new()
    {
        return Map<TSrc, TDst>;
    }

    private void Map<TSrc, TDst>(TSrc arg1, TDst arg2)
        where TSrc : class, new() where TDst : class, new()
    {
    }
}