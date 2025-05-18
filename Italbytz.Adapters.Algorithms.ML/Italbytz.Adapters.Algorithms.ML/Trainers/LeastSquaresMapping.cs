using System;

namespace Italbytz.ML.Trainers;

public class LeastSquaresMapping(double[]? parameters)

{
    public Action<TSrc, TDst> GetMapping<TSrc, TDst>()
        where TSrc : class, new() where TDst : class, new()
    {
        return Map<TSrc, TDst>;
    }

    private void Map<TSrc, TDst>(TSrc arg1, TDst arg2)
        where TSrc : class, new() where TDst : class, new()
    {
        var score = parameters![0];
        if (arg1 is not ICustomMappingInput input)
            throw new ArgumentException(
                "The input is not of type ICustomMappingInputSchema");
        var features = input.Features;
        for (var i = 1; i < parameters.Length; i++)
            score += parameters[i] * features[i - 1];
        if (arg2 is IRegressionOutput output)
            output.Score = (float)score;
        else
            throw new ArgumentException(
                "The destination is not of type IRegressionOutput");
    }
}