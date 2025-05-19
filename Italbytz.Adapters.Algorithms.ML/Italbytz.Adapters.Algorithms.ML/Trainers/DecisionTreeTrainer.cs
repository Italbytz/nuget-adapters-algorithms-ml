using Italbytz.AI.Learning;
using Italbytz.AI.Learning.Learners;

namespace Italbytz.ML.Trainers;

public abstract class
    DecisionTreeTrainer<TOutput> : CustomClassificationTrainer<TOutput>
    where TOutput : class, new()
{
    protected readonly ILearner _learner;

    protected DecisionTreeTrainer()
    {
        _learner = new DecisionTreeLearner();
    }
}