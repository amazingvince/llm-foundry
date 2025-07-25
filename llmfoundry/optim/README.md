# Using PyTorch Optimizers in LLM Foundry

## Overview

LLM Foundry now supports all PyTorch optimizers out of the box! These are registered with a `pytorch_` prefix followed by the lowercase optimizer name.

## Naming Convention

PyTorch optimizer names are registered as:
- `torch.optim.Adam` → `pytorch_adam`
- `torch.optim.AdamW` → `pytorch_adamw`
- `torch.optim.SGD` → `pytorch_sgd`
- `torch.optim.RMSprop` → `pytorch_rmsprop`
- etc.

## Usage in YAML

### Basic Example with AdamW

```yaml
optimizer:
  name: pytorch_adamw
  lr: 0.001
  betas: [0.9, 0.999]
  eps: 1e-08
  weight_decay: 0.01
```

### Common PyTorch Optimizers

#### Adam
```yaml
optimizer:
  name: pytorch_adam
  lr: 0.001
  betas: [0.9, 0.999]
  eps: 1e-08
  weight_decay: 0
  amsgrad: false
```

#### SGD with Momentum
```yaml
optimizer:
  name: pytorch_sgd
  lr: 0.1
  momentum: 0.9
  weight_decay: 0.0001
  dampening: 0
  nesterov: true
```

#### RMSprop
```yaml
optimizer:
  name: pytorch_rmsprop
  lr: 0.01
  alpha: 0.99
  eps: 1e-08
  weight_decay: 0
  momentum: 0
  centered: false
```

#### AdaGrad
```yaml
optimizer:
  name: pytorch_adagrad
  lr: 0.01
  lr_decay: 0
  weight_decay: 0
  initial_accumulator_value: 0
  eps: 1e-10
```

#### AdaDelta
```yaml
optimizer:
  name: pytorch_adadelta
  lr: 1.0
  rho: 0.9
  eps: 1e-06
  weight_decay: 0
```

#### NAdam
```yaml
optimizer:
  name: pytorch_nadam
  lr: 0.002
  betas: [0.9, 0.999]
  eps: 1e-08
  weight_decay: 0
  momentum_decay: 0.004
```

#### RAdam
```yaml
optimizer:
  name: pytorch_radam
  lr: 0.001
  betas: [0.9, 0.999]
  eps: 1e-08
  weight_decay: 0
```

## Available PyTorch Optimizers

The following PyTorch optimizers are automatically registered:
- `pytorch_adam` - Adaptive Moment Estimation
- `pytorch_adamw` - Adam with decoupled weight decay
- `pytorch_sgd` - Stochastic Gradient Descent
- `pytorch_rmsprop` - RMSprop
- `pytorch_adagrad` - Adaptive Gradient
- `pytorch_adadelta` - Adaptive Delta
- `pytorch_adamax` - Adamax (variant of Adam based on infinity norm)
- `pytorch_asgd` - Averaged Stochastic Gradient Descent
- `pytorch_rprop` - Resilient Backpropagation
- `pytorch_nadam` - NAdam (Adam with Nesterov momentum)
- `pytorch_radam` - Rectified Adam
- `pytorch_sparseadam` - Sparse Adam (for sparse tensors)

**Note**: LBFGS is not supported as it requires a closure function which doesn't fit the standard optimizer interface.

## Comparison with LLM Foundry Optimizers

LLM Foundry also provides specialized optimizers:

| LLM Foundry Optimizer | PyTorch Equivalent | Key Differences |
|----------------------|-------------------|-----------------|
| `decoupled_adamw` | `pytorch_adamw` | Same algorithm, composer implementation |
| `decoupled_lionw` | - | Lion optimizer with decoupled weight decay |
| `adalr_lion` | - | Lion with adaptive layer-wise learning rates |
| `clip_lion` | - | Lion with gradient clipping |

## Full Training Configuration Example

```yaml
# Train configuration using PyTorch AdamW
run_name: my_model_pytorch_adamw
model:
  name: hf_causal_lm
  pretrained_model_name_or_path: mosaicml/mpt-7b
  config_overrides:
    max_seq_len: 2048

# Using PyTorch AdamW optimizer
optimizer:
  name: pytorch_adamw
  lr: 0.00001
  betas: [0.9, 0.999]
  eps: 1e-08
  weight_decay: 0.01

# Learning rate scheduler
scheduler:
  name: cosine_with_warmup
  t_warmup: 100ba
  alpha_f: 0.1

# Training parameters
max_duration: 10ep
eval_interval: 500ba
global_train_batch_size: 128
device_train_microbatch_size: 8

# ... rest of configuration
```

## Tips and Best Practices

1. **Parameter Names**: Use the exact parameter names from PyTorch documentation (e.g., `betas` not `beta`)

2. **Default Values**: If you don't specify a parameter, PyTorch's default will be used

3. **Learning Rate**: Always specify `lr` as it's required by all optimizers

4. **Weight Decay**: PyTorch's `Adam` uses L2 penalty while `AdamW` uses decoupled weight decay

5. **Validation**: The YAML parameters are passed directly to the PyTorch optimizer, so refer to [PyTorch optimizer documentation](https://pytorch.org/docs/stable/optim.html) for valid parameter ranges

## Troubleshooting

### Common Issues

1. **Invalid parameter error**: Check that parameter names match PyTorch's exactly
   ```yaml
   # Wrong
   optimizer:
     name: pytorch_adam
     learning_rate: 0.001  # ❌ Should be 'lr'
   
   # Correct
   optimizer:
     name: pytorch_adam
     lr: 0.001  # ✅
   ```

2. **Optimizer not found**: Ensure you're using the `pytorch_` prefix
   ```yaml
   # Wrong
   optimizer:
     name: adam  # ❌ Missing prefix
   
   # Correct
   optimizer:
     name: pytorch_adam  # ✅
   ```

## Checking Available Optimizers

To see all available optimizers in your environment:

```python
from llmfoundry.registry import optimizers

# List all registered optimizers
print("Available optimizers:")
for name in sorted(optimizers._registry.keys()):
    print(f"  - {name}")

# Filter PyTorch optimizers
pytorch_optimizers = [name for name in optimizers._registry.keys() if name.startswith('pytorch_')]
print(f"\nPyTorch optimizers: {pytorch_optimizers}")
```


