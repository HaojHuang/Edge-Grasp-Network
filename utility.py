import numpy as np

def downsample_points(pts, K):
    # if num_pts > 8K use farthest sampling
    # else use random sampling
    if pts.shape[0] >= 2*K:
        sampler = FarthestSampler()
        return sampler(pts, K)
    else:
        return pts[np.random.choice(pts.shape[0], K, replace=(K<pts.shape[0])), :]


class FarthestSampler:
    def __init__(self):
        pass

    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)

    def __call__(self, pts, k):
        index_list = []
        farthest_pts = np.zeros((k, 3), dtype=np.float32)
        index = np.random.randint(len(pts))
        farthest_pts[0] = pts[index]
        index_list.append(index)
        distances = self._calc_distances(farthest_pts[0], pts)
        for i in range(1, k):
            index = np.argmax(distances)
            farthest_pts[i] = pts[index]
            index_list.append(index)
            distances = np.minimum(
                distances, self._calc_distances(farthest_pts[i], pts))
        return farthest_pts,index_list