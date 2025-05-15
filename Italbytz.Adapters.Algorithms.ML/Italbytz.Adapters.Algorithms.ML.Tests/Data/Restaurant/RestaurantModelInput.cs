using Microsoft.ML.Data;

namespace Italbytz.ML.Tests.Data;

public class RestaurantModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"alternate")]
    public float Alternate { get; set; }

    [LoadColumn(1)] [ColumnName(@"bar")] public float Bar { get; set; }

    [LoadColumn(2)]
    [ColumnName(@"fri/sat")]
    public float Fri_sat { get; set; }

    [LoadColumn(3)]
    [ColumnName(@"hungry")]
    public float Hungry { get; set; }

    [LoadColumn(4)]
    [ColumnName(@"patrons")]
    public float Patrons { get; set; }

    [LoadColumn(5)] [ColumnName(@"price")] public float Price { get; set; }

    [LoadColumn(6)]
    [ColumnName(@"raining")]
    public float Raining { get; set; }

    [LoadColumn(7)]
    [ColumnName(@"reservation")]
    public float Reservation { get; set; }

    [LoadColumn(8)] [ColumnName(@"type")] public float Type { get; set; }

    [LoadColumn(9)]
    [ColumnName(@"wait_estimate")]
    public float Wait_estimate { get; set; }

    [LoadColumn(10)]
    [ColumnName(@"will_wait")]
    public uint Will_wait { get; set; }
}