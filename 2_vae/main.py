import jax
import jax.numpy as jnp
import optax

import equinox as eqx
import equinox.nn as nn
import numpy as np

from datasets import load_dataset
from PIL import Image
from typing import Tuple
from tqdm import tqdm


input_dim = 784
hidden_dim = 200
latent_dim = 20
epochs = 30
learning_rate = 3e-4
batch_size = 1024


class Encoder(eqx.Module):
    linear: nn.Linear
    linear_mu: nn.Linear
    linear_logvar: nn.Linear

    def __init__(self, key: jax.random.PRNGKey):
        k1, k2, k3 = jax.random.split(key, 3)

        self.linear = nn.Linear(input_dim, hidden_dim, key=k1)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim, key=k2)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim, key=k3)

    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        hidden = self.linear(x)
        hidden = jax.nn.relu(hidden)

        mu = self.linear_mu(hidden)
        logvar = self.linear_logvar(hidden)
        sigma = jnp.exp(0.5 * logvar)

        return mu, sigma


class Decoder(eqx.Module):
    layers: list

    def __init__(self, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key, 2)

        self.layers = [
            nn.Linear(latent_dim, hidden_dim, key=k1),
            jax.nn.relu,
            nn.Linear(hidden_dim, input_dim, key=k2),
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = layer(x)
        return jax.nn.sigmoid(x)


def reparameterize(mu: jnp.ndarray, sigma: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
    eps = jax.random.normal(key, mu.shape)
    return mu + sigma * eps


class VAE(eqx.Module):
    encoder: Encoder
    decoder: Decoder

    def __init__(self, key: jax.random.PRNGKey):
        k1, k2 = jax.random.split(key, 2)
        self.encoder = Encoder(k1)
        self.decoder = Decoder(k2)

    def calc_loss(self, x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        mu, sigma = jax.vmap(self.encoder)(x)
        z = reparameterize(mu, sigma, key)
        x_hat = jax.vmap(self.decoder)(z)

        recon_loss = jnp.mean(jnp.sum(jnp.square(x - x_hat), axis=-1))
        kl_loss = jnp.mean(-jnp.sum(1 + jnp.log(sigma ** 2) - mu ** 2 - sigma ** 2, axis=-1))
        return recon_loss + kl_loss

    def decode(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)


@jax.jit
def preprocess(x: jnp.ndarray) -> jnp.ndarray:
    x = x.astype(jnp.float32) / 255.
    x = x.reshape((-1, input_dim))
    return x


@jax.jit
def postprocess(x: jnp.ndarray) -> jnp.ndarray:
    x = x.reshape((-1, 28, 28))
    x = (x * 255).astype(jnp.uint8)
    return x


def main():
    key = jax.random.PRNGKey(42)

    # prepare the VAE model
    key, model_key = jax.random.split(key)
    model = VAE(model_key)
    print("[Model]")
    print(model)

    # prepare training
    opt = optax.adamw(learning_rate)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    dataset = load_dataset("mnist")
    dataset = dataset.with_format("jax")

    key, train_key = jax.random.split(key)

    # training
    @eqx.filter_jit
    def train_step(model: VAE, opt_state: optax.OptState, x: jnp.ndarray, key: jax.random.PRNGKey) -> Tuple[VAE, optax.OptState, jnp.ndarray]:
        loss_fn = lambda model: model.calc_loss(x, key)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        
        updates, opt_state = opt.update(
            grads, opt_state, eqx.filter(model, eqx.is_array)
        )
        
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    for epoch in range(epochs):
        total_loss = 0.
        count = 0
        total_batches = len(dataset["train"]) // batch_size
        for batch in (pbar := tqdm(dataset["train"].shuffle().iter(batch_size, drop_last_batch=True), total=total_batches)):
            x = batch["image"]
            x = preprocess(x)

            train_key, subrng = jax.random.split(train_key)
            model, opt_state, loss = train_step(model, opt_state, x, subrng)

            total_loss += loss
            count += 1
            pbar.set_description(f"Loss: {total_loss / count:.4f}")
        print(f"Epoch {epoch + 1} - Loss: {total_loss / count}")

    # generate samples
    key, generation_key = jax.random.split(key)
    z = jax.random.normal(generation_key, (10, latent_dim))
    x_hat = jax.vmap(model.decode)(z)
    x_hat = postprocess(x_hat)

    x_hat_np = np.array(x_hat)
    x_hat_np = x_hat_np.reshape(2, 5, 28, 28).transpose(0, 2, 1, 3).reshape(56, 140)
    Image.fromarray((x_hat_np * 255).astype(np.uint8)).save("generated_samples.png")



if __name__ == "__main__":
    main()
