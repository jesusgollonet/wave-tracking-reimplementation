import math


def calculate_inertia_ratio(moments):
    # Calculate the denominator using the correct normalization
    denominator = math.sqrt(
        (2 * moments["m11"]) ** 2 + (moments["m20"] - moments["m02"]) ** 2
    )

    # Small epsilon to avoid division by zero
    epsilon = 0.01
    if denominator < epsilon:
        return 0.0  # handle division by zero or near-zero

    # Calculate the sin and cos of the angle
    cosmin = (moments["m20"] - moments["m02"]) / denominator
    sinmin = 2 * moments["m11"] / denominator
    cosmax = -cosmin
    sinmax = -sinmin

    # Calculate the minimum and maximum inertia
    imin = (
        0.5 * (moments["m20"] + moments["m02"])
        - 0.5 * (moments["m20"] - moments["m02"]) * cosmin
        - moments["m11"] * sinmin
    )

    imax = (
        0.5 * (moments["m20"] + moments["m02"])
        - 0.5 * (moments["m20"] - moments["m02"]) * cosmax
        - moments["m11"] * sinmax
    )

    ratio = imin / imax
    return ratio
