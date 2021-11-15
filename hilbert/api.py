import math

from hilbertcurve.hilbertcurve import HilbertCurve

p = 2
n = 2
hilbert_curve = HilbertCurve(p, n)
# distances = list(range(int(math.pow(2, n * p))))  # 2 ^ (np)
# points = hilbert_curve.points_from_distances(distances)
# for point, dist in zip(points, distances):
#     print(f'point(h={dist}) = {point}')

points = [[0, 0], [1, 1], [1, 0]]
distances = hilbert_curve.distances_from_points(points, match_type=True)
for point, dist in zip(points, distances):
    print(f'distance(x={point}) = {dist}')
