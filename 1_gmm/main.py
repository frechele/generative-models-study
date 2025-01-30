import numpy as np

import matplotlib.pyplot as plt


def multivariate_normal(x, mean, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    d = len(x)
    z = 1. / (np.sqrt((2 * np.pi) ** d * det))
    y = z * np.exp(-0.5 * (x - mean).T @ inv @ (x - mean))
    return y


def gmm(x, phis, means, covs):
    y = 0
    for phi, mean, cov in zip(phis, means, covs):
        y += phi * multivariate_normal(x, mean, cov)
    return y


def likelihood(xs, phis, means, covs):
    L = 0
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, means, covs)
        L += np.log(y + 1e-8)
    return L / N


def sample_gmm(phis, means, covs, n_samples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    def sample():
        z = rng.choice(len(phis), p=phis)
        return rng.multivariate_normal(means[z], covs[z])
    return np.array([sample() for _ in range(n_samples)])


def generate_dataset(n_samples: int, seed: int) -> np.ndarray:
    mus = np.array([[2.0, 54.5],
                     [4.3, 80.0]])
    covs = np.array([[[0.07, 0.44],
                       [0.44, 33.7]],
                       [[0.17, 0.94],
                        [0.94, 36.0]]])
    phis = np.array([0.35, 0.65])

    return sample_gmm(phis, mus, covs, n_samples, seed)


def main():
    # Generate Dataset
    N = 500
    xs = generate_dataset(N, 42)

    plt.scatter(xs[:, 0], xs[:, 1], s=0.5)
    plt.savefig("dataset.png")

    # Initialize Parameters
    K = 2
    phis = np.ones(K) / K
    means = np.array([[0.0, 50.0], [0.0, 100.0]])
    covs = np.array([np.eye(2), np.eye(2)])

    # Update Parameters
    MAX_ITERS = 10
    TOLERANCE = 1e-4

    cur_likelihood = likelihood(xs, phis, means, covs)
    prev_likelihood = cur_likelihood
    print(f"Initial likelihood: {cur_likelihood:.4f}")

    for iter in range(MAX_ITERS):
        # E-step
        qs = np.zeros((N, K))
        for n, x in enumerate(xs):
            for k, (phi, mean, cov) in enumerate(zip(phis, means, covs)):
                qs[n, k] = phi * multivariate_normal(x, mean, cov)
            qs[n] /= gmm(x, phis, means, covs)

        # M-step
        qs_sum = np.sum(qs, axis=0)
        for k in range(K):
            # 1. phis
            phis[k] = qs_sum[k] / N

            # 2. means
            c = 0
            for n in range(N):
                c += qs[n, k] * xs[n]
            means[k] = c / qs_sum[k]

            # 3. covs
            c = 0
            for n in range(N):
                diff = xs[n] - means[k]
                diff = diff[:, np.newaxis]
                c += qs[n, k] * diff @ diff.T
            covs[k] = c / qs_sum[k]

        # Evaluation
        cur_likelihood = likelihood(xs, phis, means, covs)
        print(f"Iteration {iter + 1} likelihood: {cur_likelihood:.4f}")

        if np.abs(cur_likelihood - prev_likelihood) < TOLERANCE:
            break
        prev_likelihood = cur_likelihood

    print(f"Converged in {iter + 1} iterations")
    print(f"Final likelihood: {cur_likelihood:.4f}")
    print("Phis: ", phis)
    print("Means: ", means)
    print("Covs: ", covs)

    # Plot
    plt.cla()
    plt.scatter(xs[:, 0], xs[:, 1], c="red", label="Data")
    pred_xs = sample_gmm(phis, means, covs, N, 12345)
    plt.scatter(pred_xs[:, 0], pred_xs[:, 1], alpha=0.5, c="blue", label="Generated")
    plt.legend()
    plt.savefig("final.png")


if __name__ == "__main__":
    main()
