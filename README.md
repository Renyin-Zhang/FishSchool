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
This project includes a sequential C program, a multi-threaded and distributed implementation using OpenMP, integrated with MPI.

- **Sequential Program**: Simulates fish school behavior in a grid, focusing on optimizing an objective function.
- **Parallel Implementation with OpenMP & MPI**: Utilizes OpenMP for parallelizing the algorithm on a single machine with multiple cores, runs across multiple nodes using MPI for data distribution and collection.

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
