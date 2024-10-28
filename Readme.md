# Optimization Project

## Overview
This project is a requirement for our coursework, CISC 820: Quantitative Foundations, for the first year of the Ph.D. program in Computing and Information Sciences at RIT. The project implements optimization techniques using three different methods viz. Gradient Descent, Newton's method and Quasi-Netwon method to solve three given objective functions. We also implemented `Adam` as bonus method for optimization. The core logic is in the `main.py` script inside `src` folder, and outputs, including graphs and results are stored in `curves` and `results` subfolders respectively inside the `artifacts` folder.

## Table of Contents
- [Installation](#installation)
- [Running the Project](#running-the-project)
- [Authors](#authors)
- [License](#license)

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/SrjPdl/Optimization_QF.git
    cd Optimization_QF
    ```

2. **Create a virtual environment**:
    Create a virtual environment to isolate the project’s dependencies.
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On **Windows**:
        ```bash
        venv\Scripts\activate
        ```
    - On **macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```

4. **Install dependencies**:
    Install the necessary packages from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project
To run the project, execute the `main.py` script. Ensure that your virtual environment is active.
 ```bash
 python3 src/main.py
 ```
 # Authors
- Suraj Poudel
- Charles Devlen

# License
This project is licensed under the [MIT](https://www.mit.edu/~amini/LICENSE.md) License. 

