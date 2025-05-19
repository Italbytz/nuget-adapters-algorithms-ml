using Italbytz.AI.Learning;
using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace Italbytz.ML.Trainers;

/// <inheritdoc />
public class DecisionTreeBinaryTrainer : DecisionTreeTrainer<
    BinaryClassificationOutput>
{
    private IDataExcerpt? _dataExcerpt;
    private IDataSetSpecification? _spec;

    protected override void PrepareForFit(IDataView input)
    {
        _dataExcerpt = input.GetDataExcerpt();
        _spec = _dataExcerpt.GetDataSetSpecification();
        var dataSet = _dataExcerpt.GetDataSet(_spec);
        _learner.Train(dataSet);
    }

    protected override
        CustomMappingEstimator<ClassificationInput, BinaryClassificationOutput>
        GetCustomMappingEstimator()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var mapping = new DecisionTreeMapping(_learner, _dataExcerpt, _spec);
        return mlContext.Transforms
            .CustomMapping(
                mapping
                    .GetMapping<ClassificationInput,
                        BinaryClassificationOutput>(), null);
    }
}