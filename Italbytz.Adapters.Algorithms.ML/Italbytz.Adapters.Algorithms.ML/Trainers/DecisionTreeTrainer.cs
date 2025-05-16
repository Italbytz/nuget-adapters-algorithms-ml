using Italbytz.AI.Learning;
using Italbytz.AI.Learning.Learners;

namespace Italbytz.ML.Trainers;

public abstract class
    DecisionTreeTrainer<TInput, TOutput> : CustomTrainer<TInput, TOutput>
    where TOutput : class, new() where TInput : class, new()
{
    protected readonly ILearner _learner;

    protected DecisionTreeTrainer()
    {
        _learner = new DecisionTreeLearner();
    }
}