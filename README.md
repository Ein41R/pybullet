### dir structure explanation

pybullet/
├── .github/                  # GitHub workflows
│   └── workflows/
├── docs/                     # Documentation
│   ├── architecture.md
│   └── setup.md
├── src/                      # Main source code
│   ├── rl/                   # Reinforcement learning core
│   │   ├── agents/           # Agent implementations
│   │   ├── envs/             # Custom environments
│   │   ├── models/           # Neural network architectures
│   │   ├── utils/            # Helper functions
│   │   └── __init__.py
│   ├── ros/                  # ROS integration
│   │   ├── interfaces/       # ROS interfaces
│   │   ├── launch/           # Launch files
│   │   └── nodes/            # ROS nodes
│   └── __init__.py
├── tests/                    # Unit and integration tests
│   ├── rl/
│   └── ros/
├── scripts/                  # Utility scripts
│   ├── setup_environment.sh
│   └── run_experiments.py
├── configs/                  # Configuration files
│   ├── hyperparameters.yaml
│   └── ros/
├── data/                     # Data files (gitignored)
│   ├── raw/
│   ├── processed/
│   └── models/
├── requirements/             # Dependency specifications
│   ├── base.txt              # Core dependencies
│   ├── dev.txt               # Development dependencies
│   └── ros.txt               # ROS-specific dependencies
├── .gitignore
├── pyproject.toml            # Modern Python project config
├── README.md
└── LICENSE