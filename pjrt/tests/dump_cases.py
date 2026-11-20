import jax, jax.numpy as jnp
import numpy as np

x = np.arange(6., dtype=np.float32).reshape(2, 3)
w = np.arange(12., dtype=np.float32).reshape(3, 4) / 10
b = np.arange(4., dtype=np.float32)

cases = {
    "mlp_forward": (lambda x, w: jnp.sum(jax.nn.relu(x @ w)), (x, w)),
    "mlp_grad_x": (lambda x, w: jnp.sum(jax.grad(
        lambda y: jnp.sum(jax.nn.relu(y @ w)))(x)), (x, w)),
    "mlp_grad_w": (lambda x, w: jnp.sum(jax.grad(
        lambda y: jnp.sum(jax.nn.relu(x @ y)))(w)), (x, w)),
    "add": (lambda x, w: jnp.sum(x + x), (x, w)),
    "mul": (lambda x, w: jnp.sum(x * x), (x, w)),
    "sub": (lambda x, w: jnp.sum(x - x), (x, w)),
    "max": (lambda x, w: jnp.sum(jnp.maximum(x, 0.5)), (x, w)),
    "neg": (lambda x, w: jnp.sum(-x), (x, w)),
    "relu": (lambda x, w: jnp.sum(jax.nn.relu(x)), (x, w)),
    "reshape_bcast": (lambda x, w: jnp.sum(
        jnp.broadcast_to(x.reshape(2, 3, 1), (2, 3, 4))), (x, w)),
    "convert": (lambda x, w: jnp.sum(
        x.astype(jnp.float32) + 0, dtype=jnp.float32), (x, w)),
    "dot_bt": (lambda x, w: jnp.sum(x @ w.T[:3, :3]), (x, w)),
    "two_ops": (lambda x, w: jnp.sum(jax.nn.relu(x @ w) + b), (x, w)),
}

for name, (fn, args) in cases.items():
    try:
        jax.jit(fn).lower(*args).compile()
        print("compiled", name)
    except Exception as e:
        print("FAILED", name, type(e).__name__, str(e)[:300])
