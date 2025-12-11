import jax
import jax.numpy as jnp

def encode_board(state):
    """
    state: (28,) int8 JAX array
    returns: (24, 9) float32 JAX array
    """

    # Points are state[1:25], shape (24,)
    pts = state[1:25]

    # plane 0: empty
    empty = (pts == 0).astype(jnp.float32)

    # white planes
    white = pts > 0
    w1 = (pts == 1).astype(jnp.float32) * white
    w2 = (pts == 2).astype(jnp.float32) * white
    w3 = (pts == 3).astype(jnp.float32) * white
    w4 = (pts >= 4).astype(jnp.float32) * white

    # black planes
    neg = -pts
    black = pts < 0
    b1 = (neg == 1).astype(jnp.float32) * black
    b2 = (neg == 2).astype(jnp.float32) * black
    b3 = (neg == 3).astype(jnp.float32) * black
    b4 = (neg >= 4).astype(jnp.float32) * black

    # stack into (24, 9)
    return jnp.stack([empty,
                      w1, w2, w3, w4,
                      b1, b2, b3, b4], axis=1)

def extract_aux(state):
    w_bar = state[0]
    b_bar = state[25]
    w_off = state[26]
    b_off = state[27]

    return jnp.array([
        (w_bar > 0).astype(jnp.float32),
        w_bar / 15.0,
        w_off / 15.0,
        (b_bar > 0).astype(jnp.float32),
        b_bar / 15.0,
        b_off / 15.0
    ], dtype=jnp.float32)

encode_board_batch = jax.vmap(encode_board)
extract_aux_batch  = jax.vmap(extract_aux)
