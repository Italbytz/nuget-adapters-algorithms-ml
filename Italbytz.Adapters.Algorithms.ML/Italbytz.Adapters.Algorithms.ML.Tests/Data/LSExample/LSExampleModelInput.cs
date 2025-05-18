using Microsoft.ML.Data;

namespace Italbytz.ML.Tests.Data.LSExample;

public class LSExampleModelInput
{
    [LoadColumn(0)] [ColumnName(@"x1")] public float X1 { get; set; }

    [LoadColumn(1)] [ColumnName(@"x2")] public float X2 { get; set; }

    [LoadColumn(2)] [ColumnName(@"y")] public float Y { get; set; }
}