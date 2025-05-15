using Microsoft.ML.Data;

namespace logicGP.Tests.Data.Real;

public class CarEvaluationModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"buying")]
    public string Buying { get; set; }

    [LoadColumn(1)] [ColumnName(@"maint")] public string Maint { get; set; }

    [LoadColumn(2)] [ColumnName(@"doors")] public string Doors { get; set; }

    [LoadColumn(3)]
    [ColumnName(@"persons")]
    public string Persons { get; set; }

    [LoadColumn(4)]
    [ColumnName(@"lug_boot")]
    public string Lug_boot { get; set; }

    [LoadColumn(5)]
    [ColumnName(@"safety")]
    public string Safety { get; set; }

    [LoadColumn(6)] [ColumnName(@"class")] public string Class { get; set; }
}