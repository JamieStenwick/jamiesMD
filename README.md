# jamiesMD

> [Brief one- or two-sentence description of jamiesMD.]

---

## Table of Contents

1. [Installation, Compilation, and Usage](#installation-compilation-usage)
2. [Energy Comparison with HOOMD-blue](#energy-comparison)
   - [Potential Energy Comparison](#potential-energy-comparison)
   - [Total Energy Comparison](#total-energy-comparison)
3. [SAXS Comparison with HOOMD-blue](#saxs-comparison)
4. [Current Assumptions and Limitations](#assumptions-limitations)
5. [Future Implementation and Improvements](#future-improvements)
6. [Generative AI Assistance](#generative-ai-assistance)

---

<a id="installation-compilation-usage"></a>

## 1. Installation, Compilation, and Usage

### Requirements

- CUDA-capable NVIDIA GPU
- CUDA toolkit 13.x
- C++ compiler with C++23 support and a compatible NVCC version
- CMake 3.x or newer
- mathdx for cuRANDdx (already included)

### Installation

```bash
# Installation/setup commands
```

### Compilation

```bash
# Compilation commands
```

### Usage

```bash
# Example command
```

### Command-Line Arguments

| Argument | Description | Units | Example |
|---|---|---|---|
| `[argument]` | [description] | [units] | `[value]` |
| `[argument]` | [description] | [units] | `[value]` |

### Example Run

```bash
# Complete example command
```

[Optional explanation of expected output files/directories.]

---

<a id="energy-comparison"></a>

## 2. Energy Comparison with HOOMD-blue

[Describe the simulation conditions used for the comparison.]

### Comparison Conditions

| Parameter | jamiesMD | HOOMD-blue |
|---|---:|---:|
| Number of particles | [value] | [value] |
| Time step | [value] | [value] |
| Temperature | [value] | [value] |
| Potential | [value] | [value] |
| [Parameter] | [value] | [value] |

<a id="potential-energy-comparison"></a>

### 2.1 Potential Energy Comparison

[Briefly describe how the potential-energy comparison was performed.]

![jamiesMD and HOOMD-blue potential energy comparison](path/to/potential_energy_comparison.png)

*Figure 1. [Caption describing the comparison between the jamiesMD and HOOMD-blue potential energy trajectories, including any relevant simulation conditions.]*

[Optional discussion of agreement, differences, numerical error, equilibration behavior, etc.]

<a id="total-energy-comparison"></a>

### 2.2 Total Energy Comparison

[Briefly describe how the total-energy comparison was performed.]

![jamiesMD and HOOMD-blue total energy comparison](path/to/total_energy_comparison.png)

*Figure 2. [Caption describing the comparison between the jamiesMD and HOOMD-blue total energy trajectories, including any relevant simulation conditions.]*

[Optional discussion of agreement, differences, drift, fluctuations, or other observations.]

---

<a id="saxs-comparison"></a>

## 3. SAXS Comparison with HOOMD-blue

[Describe how SAXS was calculated from the jamiesMD and HOOMD-blue trajectories and the conditions used for the comparison.]

![jamiesMD and HOOMD-blue SAXS comparison](path/to/saxs_comparison.png)

*Figure 3. [Caption describing the comparison between the jamiesMD and HOOMD-blue SAXS curves.]*

### Mean Squared Error

The mean squared error between the SAXS curves is:

\[
\mathrm{MSE} = \text{[value]}
\]

[Optional explanation of how the MSE was calculated, including the q-range, normalization procedure, interpolation, or other processing.]

---

<a id="assumptions-limitations"></a>

## 4. Current Assumptions and Limitations

The current implementation of jamiesMD makes the following assumptions and has the following limitations:

- **[Assumption or limitation]**
  - [Description]
  - [Reason for the assumption, if relevant]
  - [Consequences for the simulation]

- **[Assumption or limitation]**
  - [Description]

- **[Assumption or limitation]**
  - [Description]

### Physical Model Assumptions

- [Assumption]
- [Assumption]
- [Assumption]

### Numerical Assumptions

- [Assumption]
- [Assumption]
- [Assumption]

### Implementation Limitations

- [Limitation]
- [Limitation]
- [Limitation]

---

<a id="future-improvements"></a>

## 5. Future Implementation and Improvements

The following features and improvements are currently the most relevant next steps for jamiesMD.

### High Priority

1. **[Feature / improvement]**
   - [Description]
   - [Why it is important]
   - [Potential implementation approach]

2. **[Feature / improvement]**
   - [Description]
   - [Why it is important]
   - [Potential implementation approach]

3. **[Feature / improvement]**
   - [Description]
   - [Why it is important]
   - [Potential implementation approach]

### Additional Improvements

- [Improvement]
- [Improvement]
- [Improvement]

### Longer-Term Goals

- [Goal]
- [Goal]
- [Goal]

---

<a id="generative-ai-assistance"></a>

## 6. Generative AI Assistance

Generative AI tools were used during the development of jamiesMD in the following areas.

### Tools Used

- **[Tool / model]**
  - [How it was used]

### AI-Assisted Components

#### [Component / File / Feature]

- **AI assistance:** [Describe exactly what assistance was provided.]
- **Human contribution:** [Describe what was designed, implemented, modified, or verified manually.]
- **Verification:** [Describe how the resulting code or approach was tested or independently verified.]

#### [Component / File / Feature]

- **AI assistance:** [Description]
- **Human contribution:** [Description]
- **Verification:** [Description]

### Conceptual / Technical Assistance

- [Topic for which AI was used to explain or discuss concepts]
- [Topic]
- [Topic]

### Code Assistance

- [Specific code, algorithm, debugging task, or implementation where AI contributed]
- [Specific code, algorithm, debugging task, or implementation where AI contributed]

### Documentation Assistance

- [Documentation or README content produced or edited with AI assistance]

### Scope of AI Use

[Provide an overall statement explaining the role of generative AI in the project and distinguishing AI-assisted work from the design, implementation, validation, and engineering decisions performed by the project author.]

---

## nvidia mathdx SDK notice

This software contains source code provided by NVIDIA Corporation.

## Contact

Reach me at jamiestenwick@gmail.com
