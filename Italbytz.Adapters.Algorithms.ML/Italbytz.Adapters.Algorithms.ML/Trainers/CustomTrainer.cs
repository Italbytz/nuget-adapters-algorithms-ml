using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace Italbytz.ML.Trainers;

public abstract class
    CustomTrainer<TInput, TOutput> : IEstimator<ITransformer>
    where TOutput : class, new()
    where TInput : class, new()
{
    /// <inheritdoc />
    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        return GetCustomMappingEstimator().GetOutputSchema(inputSchema);
    }

    /// <inheritdoc />
    public ITransformer Fit(IDataView input)
    {
        PrepareForFit(input);
        return GetCustomMappingEstimator().Fit(input);
    }

    protected abstract void PrepareForFit(IDataView input);

    protected abstract CustomMappingEstimator<TInput, TOutput>
        GetCustomMappingEstimator();
}