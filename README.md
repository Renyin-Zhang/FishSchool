# Parallel Implementation of Search Based on Fish School Behavior Using MPI and OpenMP

## Project Overview
This project extends the parallel implementation of a search algorithm inspired by Fish School Behavior (FSB) to incorporate both OpenMP and MPI for enhanced parallel processing capabilities. It aims to utilize the combined power of multi-threading and message passing for optimizing multi-dimensional optimization problems in a distributed computing environment.

## Computational Resources

This project was developed and tested on the Pawsey Supercomputer. The Pawsey Supercomputer offers advanced computing facilities that enabled us to perform high-level parallel computations and extensive simulations. 

### Prerequisites
- C compiler (GCC recommended)
- OpenMP for parallel programming support
- MPI for message passing in distributed systems

## Implementation Details

### 1. `sequential.c`
- **Purpose**: Initializes and simulates fish behavior sequentially, without parallel processing. This foundational approach focuses on simple simulation tasks such as fish movement and basic behavior.
- **Key Features**:
  - Basic data structures for representing fish.
  - Initialization routines and the main simulation loop.
  - Simple computations for functions like calculating distances.
  - Basic file input/output for recording simulation data.
  - Error handling mechanisms to manage issues during execution.

### 2. `openmp_parallel.c`
- **Purpose**: Utilizes OpenMP to parallelize the fish simulation on a single machine, making use of multiple CPU cores available in a shared-memory architecture.
- **Key Features**:
  - OpenMP directives to manage parallel loops and possibly other parallel tasks or sections.
  - Focus on reducing the overhead associated with multi-threading and optimizing memory usage among threads.
  - Designed to enhance computational efficiency on systems with multi-core processors through efficient division of simulation tasks.
 
### 3. `mpi_test.c`
- **Purpose**: Introduces MPI to distribute the fish simulation across multiple processes, aiming to enhance the scalability and handle larger datasets or more complex simulations efficiently.
- **Key Features**:
  - MPI initialization and the creation of a custom MPI datatype to handle the fish data structure efficiently across processes.
  - Distribution of initial fish data from the master node to other nodes for parallel processing.
  - Use of MPI functions such as `MPI_Scatterv` and `MPI_Gatherv` for effective data distribution and collection.

### 4. `mpi_parallel.c`
- **Purpose**: Enhances the parallelism of the fish simulation by integrating more complex behaviors such as dynamic movement and weight adjustments, using both MPI and OpenMP for distributed and multi-threaded execution.
- **Key Features**:
  - Advanced simulation functions like dynamic swimming behavior and objective-based weight adjustments.
  - Combination of MPI for inter-node communication and OpenMP for intra-node multi-threading, optimizing parallel execution.
  - Calculation of maximum objectives and updates to fish weights, showcasing a blend of MPI and OpenMP functionalities.

## Usage
Compile and run by creating a batch file `myscript`:

```c
#!/bin/sh 
#SBATCH --account=courses0101
#SBATCH --partition=debug
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:01:00
#SBATCH --exclusive
#SBATCH --mem-per-cpu=32G
#module load  openmpi/4.0.5

export OMP_NUM_THREADS=64
mpicc -fopenmp -o parallel parallel.c -lm
srun ./parallel
```

Submit the job by: `sbatch myscript`

## Experiments and Analysis
Perform experiments by varying the number of MPI processes and OpenMP threads to analyze the performance impact. Document the scalability and efficiency of the combined MPI and OpenMP implementation.
