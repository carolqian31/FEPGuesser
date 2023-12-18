import random
import numpy as np
import torch


def points_sampling(points, step_size=None, max_hop=None, vertex_num=None, sigma=None, strategy='uniform'):
    """
    sample some new points with the given strategy
    Args:
        points: batch_size start point, shape=[batch_size,dim]
        step_size: the max length one step can reach
        max_hop: the max hop one point can walk at a time
        vertex_num: the number of one point should sample
        sigma: be used to calculate covariance of gaussian distribution, covariance = sigma*I
        strategy: the strategy we use to sample
            'uniform': sample from uniform distribution where low=-step_size, high = step_size
            'gaussian' sample from gaussian distribution where mean=points, covariance=sigma*I
    Returns: the new sampled points array, shape = [batch_size, vertex_num, n]
    """
    if vertex_num == 0 or vertex_num is None:
        return None

    [batch_size, dim] = points.shape
    res = None

    if strategy == 'uniform':
        if step_size is None or max_hop is None:
            raise ValueError("if use uniform distribution, please input arguments step_size and max_hop")
        vertex_hops = np.random.choice(range(1, max_hop + 1), batch_size * vertex_num).reshape(batch_size, vertex_num)
        perturbations = np.zeros(shape=(batch_size, vertex_num, dim), dtype=np.float32)

        mask = np.ones(shape=(batch_size, vertex_num, dim, max_hop))

        for batch in range(0, batch_size):
            for vertex_idx in range(0, vertex_num):
                mask[batch, vertex_idx, :, vertex_hops[batch, vertex_idx]:] = 0
        rand_uniform_matrix = np.random.uniform(low=-step_size, high=step_size, size=(batch_size, vertex_num,
                                                                                      dim, max_hop))
        rand_uniform_matrix = np.multiply(rand_uniform_matrix, mask)
        perturbations += np.sum(rand_uniform_matrix, axis=-1).reshape(batch_size, vertex_num, dim)
        perturbations = np.transpose(perturbations, (1, 0, 2))
        res = np.array([perturb.reshape(batch_size, dim) + points for perturb in perturbations]) \
            .reshape(vertex_num, batch_size, dim)
        res = np.transpose(res, (1, 0, 2))

    elif strategy == 'gaussian':
        res = None
        if sigma is None:
            raise ValueError("if use gaussian distribution, please input arguments sigma")
        for batch in range(0, batch_size):
            point = points[batch].reshape(-1, dim) # (1,100))
            mean = point.reshape(dim)   # (100,)
            cov = np.eye(dim, dim) * sigma  # (100,100)
            new_points_size = (vertex_num[batch])
            now_batch_new_points = np.random.multivariate_normal(mean, cov, size=new_points_size)\
                .reshape(-1, new_points_size, dim)
            if res is None:
                res = now_batch_new_points
            else:
                res = np.vstack((res, now_batch_new_points))
    else:
        raise ValueError("please input the correct strategy")

    return res


def points_sampling_gpu(points, step_size=None, max_hop=None, vertex_num=None, sigma=None, strategy='uniform',
                        sigma_list=None):
    """
    sample some new points with the given strategy
    Args:
        points: batch_size start point, shape=[1,dim] tensor
        step_size: the max length one step can reach
        max_hop: the max hop one point can walk at a time
        vertex_num: the number of one point should sample
        sigma: be used to calculate covariance of gaussian distribution, covariance = sigma*I
        strategy: the strategy we use to sample
            'uniform': sample from uniform distribution where low=-step_size, high = step_size
            'gaussian' sample from gaussian distribution where mean=points, covariance=sigma*I
    Returns: the new sampled points array, shape = [sum(vertex_num), n]
    """
    if vertex_num == 0 or vertex_num is None:
        return None

    [batch_size, dim] = points.shape
    res = None
    import torch

    if strategy == 'uniform':
        if step_size is None or max_hop is None:
            raise ValueError("if use uniform distribution, please input arguments step_size and max_hop")
        for batch in range(0, batch_size):
            point = points[batch]
            perturbations = torch.zeros(vertex_num[batch], dim).to(point.device)
            mask = torch.ones(vertex_num[batch], dim, max_hop).to(point.device)
            vertex_hops = np.random.choice(range(1, max_hop + 1), vertex_num[batch]).reshape(vertex_num[batch])
            # vertex_hops = torch.randint(1, max_hop + 1, (vertex_num[batch],)).to(point.device)
            for vertex_idx in range(0, vertex_num[batch]):
                mask[vertex_idx, :, vertex_hops[vertex_idx]:] = 0
            rand_uniform_matrix = torch.rand(vertex_num[batch], dim, max_hop)\
                                      .to(point.device) * 2 * step_size - step_size
            rand_uniform_matrix = torch.mul(rand_uniform_matrix, mask)
            perturbations += torch.sum(rand_uniform_matrix, dim=-1)
            if res is None:
                res = perturbations + point
            else:
                res = torch.cat((res, perturbations + point), dim=0)

    elif strategy == 'gaussian':
        res = None
        if sigma is None and sigma_list is None:
            raise ValueError("if use gaussian distribution, please input arguments sigma")

        for batch in range(0, batch_size):
            point = points[batch]
            mean = point.reshape(dim)
            if sigma_list is not None:
                sigma = sigma_list[batch]
            if not isinstance(sigma, torch.Tensor):
                sigma = torch.tensor(sigma)
            cov = torch.eye(dim).to(point.device) * sigma.to(point.device)
            new_points_size = vertex_num[batch]
            now_batch_new_points = torch.distributions.multivariate_normal.MultivariateNormal(mean, cov)\
                .sample(sample_shape=torch.Size([new_points_size])).reshape(new_points_size, dim)
            if res is None:
                res = now_batch_new_points
            else:
                res = torch.cat((res, now_batch_new_points), dim=0)
    else:
        raise ValueError("please input the correct strategy")
    return res
