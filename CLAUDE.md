# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

From-scratch implementation of the AlphaZero reinforcement learning framework for chess using Python and PyTorch. The system learns chess entirely through self-play — no human game data required. The two core components (MCTS and a deep residual neural network) improve each other iteratively.

## Configuration

**Edit `src/config.py` before running anything:**
```python
rootDir = '/home/owensr/chess'  # root data directory
lr = 0.0003
batch_size = 3072
```

All data, model checkpoints, and output files are written under `rootDir`.

## Key commands

```bash
# Generate training data via Stockfish self-play
python3 src/generate.py <runId> <runtimeSeconds>

# De-duplicate and split game data into train/test/validate
python3 src/game_de_dup.py

# Train network for one pass through training data
python3 src/train_one.py <runId> <trainDir> <runtimeSeconds>

# Find best learning rate before a full training run
python3 src/lr_search.py

# Evaluate two networks against each other (edit script to set filenames)
python3 src/evaluator.py

# Play a single game between AlphaZero and/or Stockfish
python3 src/match.py --aType alpha --aAnetwork /path/to/best.gz --aAsteps 777 \
                     --bType stockfish --bSFHash 256 --bSFDepth 5

# Run ongoing Elo-rated tournament (edit players list in script)
python3 src/tournament.py

# Run tests
python3 src/test.py
```

## Typical training workflow

```
generate.py   → raw game data in data/games/
game_de_dup.py → deduplicated positions in data/train/, data/test/, data/validate/
train_one.py  → trains latest.gz, saves timestamped checkpoint
evaluator.py  → compare latest.gz vs best.gz, keep the winner
tournament.py → benchmark against Stockfish with Elo tracking
```

## Architecture

### Data flow

1. **`generate.py`** plays Stockfish games and records `(board_state, policy, value)` tuples as `.gz` files in `data/games/`
2. **`game_de_dup.py`** hashes positions with SHA-256 (backed by a `dbm` database) to remove duplicates, then splits 80/10/10
3. **`train_one.py`** / **`train_all_games.py`** load from `data/train/`, train the network, and save to `data/model_data/`

### Neural network (`alpha_net.py`)

`ChessNet` consists of:
- `ConvBlock`: 22-channel input → 256-filter conv + batch norm + ReLU
- 19× `ResBlock`: 256-channel residual blocks with two conv layers each
- `OutBlock`: splits into two heads:
  - **Policy head**: 256→128 conv → FC → 4,672-output softmax (8×8×73 moves)
  - **Value head**: 256→1 conv → FC → tanh scalar in [-1, +1]

Loss (`AlphaLoss`) = MSE(value) + cross-entropy(policy).

Models are saved/loaded as `compress_pickle` `.gz` files.

### Board representation (`chess_board.py` + `encoder_decoder.py`)

`chess_board.board` stores the board as an 8×8 numpy array of single-character strings. White pieces are uppercase (`R N B Q K P`), black lowercase (`r n b q k p`), empty squares are `" "`. Additional state: castling move counts per rook/king, en passant column index, move count, repetition counters.

`encode_board()` converts this to a 22-channel 8×8 tensor:
- Channels 0–11: piece presence (one-hot per piece type)
- Channels 12–16: player to move, castling rights (4 bits)
- Channels 17–21: move count, repetitions (w/b), no-progress count, en passant column

`decode_board()` reverses this for inspection. `encode_action()` / `decode_action()` map between move tuples and the 4,672-element policy vector.

### MCTS (`MCTS_chess.py`)

Implements PUCT (Polynomial Upper Confidence Trees). The neural network supplies prior probabilities at each leaf. Tree nodes store visit counts, Q-values, and priors. Self-play games are recorded as sequences of `(encoded_board, policy_vector, outcome)`.

### Player interface (`Player.py`, `AlphaZero_player.py`, `Stockfish.py`)

`Player` is an abstract base class. `AlphaZero_player` wraps MCTS + `ChessNet`. `Stockfish` wraps the Stockfish UCI engine as a subprocess. Both expose a common `move()` interface used by `match.py` and `tournament.py`.

## Data directory layout

```
<rootDir>/
├── data/
│   ├── games/        # raw self-play .gz files from generate.py
│   ├── train/        # deduplicated training positions
│   ├── test/         # test positions
│   ├── validate/     # validation positions
│   ├── model_data/
│   │   ├── latest.gz # current best net (used by train_one.py)
│   │   ├── best.gz   # evaluated best net (used by tournament.py)
│   │   └── random.gz # randomly initialised starting net
│   └── graphs/       # loss plots from training
└── stockfish/
    └── stockfish-ubuntu-x86-64-avx2  # required by generate.py
```

## Dependencies

Key packages: `torch`, `numpy`, `compress_pickle`, `matplotlib`. Full list in `src/requirements.txt` (Anaconda-era snapshot — install only what's needed rather than the full list).
