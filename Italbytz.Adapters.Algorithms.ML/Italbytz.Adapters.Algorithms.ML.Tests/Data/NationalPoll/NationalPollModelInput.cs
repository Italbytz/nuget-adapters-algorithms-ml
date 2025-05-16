using Microsoft.ML.Data;

namespace Italbytz.ML.Tests.Data.NationalPoll;

public class NationalPollModelInput
{
    [LoadColumn(0)] [ColumnName(@"Age")] public float Age { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"Physical_Health")]
    public float Physical_Health { get; set; }

    [LoadColumn(2)]
    [ColumnName(@"Mental_Health")]
    public float Mental_Health { get; set; }

    [LoadColumn(3)]
    [ColumnName(@"Dental_Health")]
    public float Dental_Health { get; set; }

    [LoadColumn(4)]
    [ColumnName(@"Employment")]
    public float Employment { get; set; }

    [LoadColumn(5)]
    [ColumnName(@"Stress_Keeps_Patient_from_Sleeping")]
    public float Stress_Keeps_Patient_from_Sleeping { get; set; }

    [LoadColumn(6)]
    [ColumnName(@"Medication_Keeps_Patient_from_Sleeping")]
    public float Medication_Keeps_Patient_from_Sleeping { get; set; }

    [LoadColumn(7)]
    [ColumnName(@"Pain_Keeps_Patient_from_Sleeping")]
    public float Pain_Keeps_Patient_from_Sleeping { get; set; }

    [LoadColumn(8)]
    [ColumnName(@"Bathroom_Needs_Keeps_Patient_from_Sleeping")]
    public float Bathroom_Needs_Keeps_Patient_from_Sleeping { get; set; }

    [LoadColumn(9)]
    [ColumnName(@"Uknown_Keeps_Patient_from_Sleeping")]
    public float Uknown_Keeps_Patient_from_Sleeping { get; set; }

    [LoadColumn(10)]
    [ColumnName(@"Trouble_Sleeping")]
    public float Trouble_Sleeping { get; set; }

    [LoadColumn(11)]
    [ColumnName(@"Prescription_Sleep_Medication")]
    public float Prescription_Sleep_Medication { get; set; }

    [LoadColumn(12)] [ColumnName(@"Race")] public float Race { get; set; }

    [LoadColumn(13)]
    [ColumnName(@"Gender")]
    public float Gender { get; set; }

    [LoadColumn(14)]
    [ColumnName(@"Number_of_Doctors_Visited")]
    public uint Number_of_Doctors_Visited { get; set; }
}