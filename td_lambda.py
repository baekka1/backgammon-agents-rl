from flax import linen as nn
from jax.numpy import jnp
import jax
import optax

from backgammon_engine import *
from backgammon_value_net import *

class CNN(nn.Module):
    kernel_size: int 7
    stride: int 1
    features: int 128

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding='SAME'
        )(x)

        return x

class ResNetv2Block(nn.Module):
    kernel_size: int 7
    stride: int 1
    features: int 128

    @nn.compact
    def __call__(self, x):
        residual = x

        # first conv
        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding='SAME'
        )(x)

        # second conv

        x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=(self.stride,),
            padding='SAME'
        )(x)

        x_out = residual + x

        return x_out

class BackgammonHead(nn.Module):
    hidden_size: int 64

    @nn.compact
    def __call__(self, resnet_out, aux_features):
        x = resent_out.mean(axis=1)

        x = jnp.concatenate([x, aux_features], axis=-1)

        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)

        x = nn.Dense(1)(x)
        x = nn.tanh(x) * 3.0

        return x

class ConvResNet1D(nn.module):
    num_blocks: int 9

    @nn.compact
    def __call__(self, x, aux_features):
        x = CNN()(x)

        for _ in range(self.num_blocks):
            x = ResNetV2Block1D()(x)

        x = BackgammonHead()(x, aux_features)

        return x
