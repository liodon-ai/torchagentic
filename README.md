# TorchAgentic

**PyTorch Model Definitions for AI Agents**

A comprehensive library of neural network architectures for building trainable AI agents, including reinforcement learning models, transformer-based agents, memory-augmented networks, and multi-agent systems.

## Features

- 🧠 **Core Architectures** - MLP, CNN, RNN/LSTM/GRU backbones
- 🎮 **RL Models** - DQN, PPO, A3C, SAC, TD3
- 🔄 **Transformers** - Decision Transformer, Perceiver IO
- 💾 **Memory Networks** - NTM, DNC (Differentiable Neural Computer)
- 👥 **Multi-Agent** - MADDPG, QMIX, VDN
- ⚡ **Utilities** - Initialization, normalization, distributions
- 🚀 **PyTorch 2.0 Compile** - torch.compile() support for 2-3x speedup

## Installation

```bash
# Basic installation
pip install torchagentic

# With transformer support
pip install torchagentic[transformers]

# Full installation
pip install torchagentic[full]

# Development installation
pip install torchagentic[dev]
```

## Quick Start

### DQN Agent

```python
import torch
from torchagentic import DQN, NatureCNN

# Create DQN model for Atari
model = DQN(
    config=ModelConfig(input_dim=4, action_dim=6),  # 4 stacked frames, 6 actions
    image_input=True,
)

# Forward pass
observations = torch.randn(32, 4, 84, 84)  # (batch, channels, height, width)
q_values = model.get_q_values(observations)

# Get action
action = model.get_action(observations, epsilon=0.1)
```

### PPO Actor-Critic

```python
from torchagentic import PPOActorCritic, ModelConfig

# Create actor-critic for continuous control
model = PPOActorCritic(
    config=ModelConfig(
        input_dim=24,      # Observation dim
        action_dim=4,      # Action dim
        hidden_dims=[256, 256],
    ),
    continuous=True,
)

# Get action and value
observation = torch.randn(1, 24)
action, log_prob, entropy, value = model.get_action_and_value(observation)
```

### Decision Transformer

```python
from torchagentic import DecisionTransformer

# Create Decision Transformer for offline RL
model = DecisionTransformer(
    config=ModelConfig(input_dim=17, action_dim=3),
    embed_dim=128,
    num_layers=3,
    max_seq_len=20,
)

# Forward with trajectory
states = torch.randn(1, 10, 17)
actions = torch.randn(1, 10, 3)
returns_to_go = torch.ones(1, 10, 1) * 100

predicted_actions = model(states, actions, returns_to_go)
```

### Neural Turing Machine

```python
from torchagentic import NeuralTuringMachine

# Create NTM
ntm = NeuralTuringMachine(
    input_size=10,
    memory_size=128,
    memory_dim=64,
    num_reads=4,
    num_writes=1,
)

# Process sequence
inputs = torch.randn(1, 50, 10)  # (batch, seq_len, input_dim)
outputs = []
hidden = None

for t in range(inputs.shape[1]):
    x = inputs[:, t:t+1, :]
    output, hidden = ntm(x, hidden)
    outputs.append(output)
```

### Multi-Agent (MADDPG)

```python
from torchagentic import MADDPGAgent

# Create MADDPG for 3 agents
model = MADDPGAgent(
    num_agents=3,
    obs_dim=10,
    action_dim=2,
    hidden_dims=[256, 256],
    shared_params=True,
)

# Get actions
observations = torch.randn(32, 3, 10)  # (batch, num_agents, obs_dim)
actions = model.get_actions(observations)

# Get centralized Q-value
q_value = model.get_q_value(observations, actions)
```

## PyTorch 2.0 Compilation

TorchAgentic provides built-in support for `torch.compile()` from PyTorch 2.0+,
enabling **2-3x speedup** for inference and training.

### Basic Compilation

```python
from torchagentic import MLPNetwork, ModelConfig

# Create model
model = MLPNetwork(ModelConfig(input_dim=64, action_dim=4))

# Compile for inference (low latency)
model.compile(mode="reduce-overhead")

# Compile for training
model.compile(mode="default", dynamic=True)

# Check if compiled
print(model.is_compiled)  # True
```

### Compilation Modes

| Mode | Use Case | Speedup |
|------|----------|---------|
| `default` | Balanced | 1.5-2x |
| `reduce-overhead` | Low latency inference | 2-3x |
| `max-autotune` | Maximum throughput | 2-4x |

### RL-Specific Optimization

```python
from torchagentic import DQN, optimize_for_inference, optimize_for_training

# Create DQN
model = DQN(ModelConfig(input_dim=4, action_dim=6), image_input=True)

# Optimize for inference (recommended for deployment)
model = optimize_for_inference(model, device="cuda")

# Optimize for training
model = optimize_for_training(model, device="cuda", batch_size=64)
```

### Benchmark Example

```python
import torch
from torchagentic import PPOActorCritic, ModelConfig

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PPOActorCritic(ModelConfig(input_dim=24, action_dim=4)).to(device)

# Compile
model.compile(mode="reduce-overhead")

# Benchmark
obs = torch.randn(1, 24, device=device)

# Uncompiled: ~0.5ms
# Compiled: ~0.2ms (2.5x speedup)
```

### Advanced Configuration

```python
from torchagentic import CompileConfig, compile_model

config = CompileConfig(
    mode="max-autotune",
    dynamic=False,
    fullgraph=True,
    backend="inductor",
    options={
        "triton.cudagraphs": True,  # Enable CUDA graphs
        "max_autotune.gemm": True,
    },
)

compiled_model = compile_model(model, config=config)
```

## Model Zoo

### Core Architectures

| Model | Description | Use Case |
|-------|-------------|----------|
| `MLPNetwork` | Multi-layer perceptron | Simple environments |
| `CNNNetwork` | Convolutional network | Visual observations |
| `NatureCNN` | Nature DQN architecture | Atari games |
| `ResNetNetwork` | Residual CNN | Complex visual tasks |
| `LSTMAgent` | LSTM-based agent | Sequential decisions |
| `GRUAgent` | GRU-based agent | Sequential decisions |

### RL Models

| Model | Algorithm | Action Space |
|-------|-----------|--------------|
| `DQN` | Deep Q-Network | Discrete |
| `DuelingDQN` | Dueling DQN | Discrete |
| `NoisyDQN` | Noisy Nets DQN | Discrete |
| `PPOActorCritic` | PPO | Both |
| `A3CNetwork` | A3C | Both |
| `SACActor` | SAC | Continuous |
| `TD3Actor` | TD3 | Continuous |

### Transformer Models

| Model | Description | Use Case |
|-------|-------------|----------|
| `TransformerAgent` | Self-attention agent | Sequential tasks |
| `DecisionTransformer` | Offline RL transformer | Offline RL |
| `PerceiverAgent` | Perceiver IO | Large inputs |

### Memory Networks

| Model | Description | Use Case |
|-------|-------------|----------|
| `NeuralTuringMachine` | NTM | Memory tasks |
| `DifferentiableNeuralComputer` | DNC | Complex memory |

### Multi-Agent

| Model | Algorithm | Cooperation |
|-------|-----------|-------------|
| `MADDPGAgent` | MADDPG | Mixed |
| `QMIXNetwork` | QMIX | Cooperative |
| `VDNNetwork` | VDN | Cooperative |

## Training Example

### PPO Training Loop

```python
import torch
from torchagentic import PPOActorCritic, ModelConfig

# Create model
model = PPOActorCritic(
    config=ModelConfig(input_dim=8, action_dim=2, hidden_dims=[64, 64]),
    continuous=False,
)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# Training step
def ppo_update(observations, actions, old_log_probs, advantages, returns):
    # Get new log probs and values
    log_probs, entropies, values = model.evaluate_actions(observations, actions)
    
    # Policy loss
    ratio = torch.exp(log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = ratio.clamp(0.8, 1.2) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    value_loss = (values - returns).pow(2).mean()
    
    # Entropy bonus
    entropy_bonus = entropies.mean() * 0.01
    
    # Total loss
    loss = policy_loss + value_loss * 0.5 - entropy_bonus
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    
    return loss.item()
```

### DQN Training Loop

```python
import torch
from torchagentic import DQN, ModelConfig
import torch.nn.functional as F

# Create models
policy_net = DQN(config=ModelConfig(input_dim=4, action_dim=2), image_input=False)
target_net = DQN(config=ModelConfig(input_dim=4, action_dim=2), image_input=False)
target_net.load_state_dict(policy_net.state_dict())

optimizer = torch.optim.Adam(policy_net.parameters(), lr=1e-4)

def dqn_update(batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones):
    # Current Q values
    q_values = policy_net(batch_states)
    q_values = q_values.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)
    
    # Target Q values
    with torch.no_grad():
        next_q_values = target_net(batch_next_states)
        next_q_max = next_q_values.max(1)[0]
        targets = batch_rewards + (1 - batch_dones) * 0.99 * next_q_max
    
    # Loss
    loss = F.mse_loss(q_values, targets)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# Update target network periodically
def update_target():
    target_net.load_state_dict(policy_net.state_dict())
```

## API Reference

### Base Classes

```python
from torchagentic import BaseAgentModel, ModelConfig

# Configuration
config = ModelConfig(
    input_dim=64,
    action_dim=4,
    hidden_dims=[256, 256],
    activation="relu",
    dropout=0.0,
)

# All models inherit from BaseAgentModel
model = YourModel(config)

# Common methods
model.forward(x)                    # Forward pass
model.get_action(obs, deterministic)  # Get action
model.get_value(obs)                # Get value estimate
model.save("path.pt")               # Save checkpoint
model.load("path.pt")               # Load checkpoint
model.get_num_params()              # Count parameters
```

### Utilities

```python
from torchagentic import orthogonal_init_, RunningNorm, DiagGaussian

# Weight initialization
orthogonal_init_(layer, gain=1.0)

# Normalization
norm = RunningNorm(256)

# Distributions
dist = DiagGaussian(mean, std)
sample = dist.sample()
log_prob = dist.log_prob(action)
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/my-feature`)
5. Create a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{torchagentic2024,
  title = {TorchAgentic: PyTorch Model Definitions for AI Agents},
  author = {Liodon AI},
  year = {2024},
  url = {https://github.com/liodon-ai/torchagentic}
}
```
