using Microsoft.ML.Data;

namespace Italbytz.ML.Tests.Data.TaxiFare;

public class TaxiModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"vendor_id")]
    public string Vendor_id { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"rate_code")]
    public float Rate_code { get; set; }

    [LoadColumn(2)]
    [ColumnName(@"passenger_count")]
    public float Passenger_count { get; set; }

    [LoadColumn(3)]
    [ColumnName(@"trip_time_in_secs")]
    public float Trip_time_in_secs { get; set; }

    [LoadColumn(4)]
    [ColumnName(@"trip_distance")]
    public float Trip_distance { get; set; }

    [LoadColumn(5)]
    [ColumnName(@"payment_type")]
    public string Payment_type { get; set; }

    [LoadColumn(6)]
    [ColumnName(@"fare_amount")]
    public float Fare_amount { get; set; }
}