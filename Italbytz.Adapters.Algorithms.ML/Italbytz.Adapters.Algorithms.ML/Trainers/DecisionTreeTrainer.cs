using Italbytz.AI.Learning;
using Italbytz.AI.Learning.Learners;
using Microsoft.ML;

namespace Italbytz.ML.Trainers;

public abstract class DecisionTreeTrainer<TSrc, TDst> : IEstimator<ITransformer>
    where TDst : class, new() where TSrc : class, new()
{
    protected readonly ILearner _learner;

    protected DecisionTreeTrainer()
    {
        _learner = new DecisionTreeLearner();
    }

    /// <inheritdoc />
    public abstract ITransformer Fit(IDataView input);

    /// <inheritdoc />
    public abstract SchemaShape GetOutputSchema(SchemaShape inputSchema);
}