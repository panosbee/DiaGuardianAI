from setuptools import setup, find_packages
import os

# Function to read the requirements.txt file
def parse_requirements(filename="requirements.txt"):
    """Load requirements from a pip requirements file."""
    try:
        with open(os.path.join(os.path.dirname(__file__), filename), 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except IOError:
        # If requirements.txt is not found, return an empty list or a default set.
        # For this project, many dependencies are heavy (torch, tensorflow, etc.)
        # and might be better handled by the user or a more detailed setup.
        # For now, let's list some core ones.
        print("Warning: requirements.txt not found. Using a minimal set of dependencies.")
        return [
            "numpy",
            "pandas",
            "PyYAML",
            "matplotlib",   # For plotting
            # "scikit-learn", # For ML models, metrics
            # "torch",        # For PyTorch based models
            # "tensorflow",   # For TensorFlow based models
            # "stable-baselines3[extra]", # For RL agents
            # "gymnasium",    # For RL agent action/observation spaces
            # "optuna",       # For hyperparameter tuning
            # "ray[tune]",    # For hyperparameter tuning
            # "seaborn",      # For plotting
            # "plotly",       # For interactive plots
            # "sentence-transformers", # For pattern repository embeddings
            # "faiss-cpu",    # Or faiss-gpu, for vector similarity search
        ]

# Read long description from README.md if it exists
try:
    with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = 'A plug and play Python library for Type 1 Diabetes management research using AI.'

setup(
    name="diaguardianai",
    version="1.0.0",
    author="DiaGuardianAI Team",
    author_email="contact@diaguardianai.com",
    description="Professional AI system for diabetes management and glucose prediction",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/diaguardianai/diaguardianai",
    project_urls={
        "Bug Tracker": "https://github.com/diaguardianai/diaguardianai/issues",
        "Documentation": "https://diaguardianai.readthedocs.io/",
        "Source Code": "https://github.com/diaguardianai/diaguardianai",
        "Clinical Demo": "https://demo.diaguardianai.com",
    },
    license="MIT",
    packages=find_packages(where=".", include=['DiaGuardianAI', 'DiaGuardianAI.*']),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=parse_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
            "ipykernel>=6.0.0"
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "clinical": [
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
            "dash>=2.14.0",
        ],
        "ml": [
            "torch>=2.0.0",
            "scikit-learn>=1.3.0",
            "scipy>=1.10.0",
        ],
        "all": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
            "pre-commit>=2.20.0",
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
            "streamlit>=1.28.0",
            "plotly>=5.15.0",
            "dash>=2.14.0",
            "torch>=2.0.0",
            "scikit-learn>=1.3.0",
            "scipy>=1.10.0",
        ],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'diaguardian-demo=DiaGuardianAI.clinical_demo.clinical_dashboard:main',
            'diaguardian-train=DiaGuardianAI.training.train_models:main',
            'diaguardian-predict=DiaGuardianAI.prediction.predict:main',
        ],
    },
    keywords=[
        "diabetes",
        "artificial intelligence",
        "machine learning",
        "glucose prediction",
        "healthcare",
        "medical ai",
        "continuous glucose monitoring",
        "insulin optimization",
    ],
)

# Note on find_packages:
# If your DiaGuardianAI package is directly in the root alongside setup.py,
# find_packages() or find_packages(where='.') should work.
# If DiaGuardianAI is in a 'src' directory, you'd use:
# package_dir={'': 'src'},
# packages=find_packages(where='src'),

# A requirements.txt file should be created with specific versions, e.g.:
# numpy>=1.20
# pandas>=1.3
# PyYAML>=5.4
# scikit-learn>=1.0
# torch>=1.10
# tensorflow>=2.7
# stable-baselines3[extra]>=1.6.0
# gymnasium>=0.26.0
# etc.