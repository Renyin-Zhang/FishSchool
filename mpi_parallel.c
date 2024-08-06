#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>
#include <omp.h>
#include <mpi.h>

#define FISH_WEIGHT     1.0     // Initial weight of the fish
#define ROUND           1000      // Number of iterations for the fish to swim
#define NUM_FISH        100000    // Total number of fish
#define MAX_WEIGHT      2*FISH_WEIGHT


// Structure definition for a fish
typedef struct {
    double x;                   // x-coordinate of fish
    double y;                   // y-coordinate of fish
    double weight;              // weight of fish
    double lastDistance;        // last computed distance from origin
} fish;

// Function to calculate distance of a fish from the origin
double calculateDistance(fish fish) {
    double x = fish.x;
    double y = fish.y;
    return sqrt(x * x + y * y);
}

// Function to calculate the objective value for a fish
double calculateObjective(fish fish, int num_fish) {
    return (calculateDistance(fish) - fish.lastDistance);
}

// Simulate the swimming behavior of the fish in parallel
void swim(fish *fishes, int start, int end) {
#pragma omp for
    for (int i = start; i < end; ++i) {
        double swimX = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
        double swimY = ((double)rand() / RAND_MAX) * 0.2 - 0.1;

        // Update the distance each step
        fishes[i].lastDistance = calculateDistance(fishes[i]);

        // Modify fish coordinates
        fishes[i].x += swimX;
        fishes[i].y += swimY;
    }
}

// Function to get the maximum objective among a subset of fish
double getMaximumObjective(fish *fishes, int num_fishes, int start, int end) {
    double max = 0;

#pragma omp parallel for reduction(max:max)  // Synchronize the max variable to prevent race condition when
    // multiple threads are trying to access and write to max variable
    for (int i = start; i < end; ++i) {
        double obj = calculateObjective(fishes[i], num_fishes);
        if (obj >= max) {
            max = obj;
        }
    }
    return max;
}

// Update the weights of a subset of fish based on their objective values in parallel
void updateWeight(fish *fishes, int num_fishes, int start, int end) {
    double local_maxObjective = getMaximumObjective(fishes, num_fishes, 0, num_fishes);
    double global_maxObjective;

    MPI_Allreduce(&local_maxObjective, &global_maxObjective, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);


#pragma omp for
    for (int i = start; i < end; ++i) {
        double obj = calculateObjective(fishes[i], num_fishes);
        fishes[i].weight += obj / global_maxObjective;
        if (fishes[i].weight >= MAX_WEIGHT){
            fishes[i].weight = MAX_WEIGHT;
        }
    }
}

// Function to calculate the barycenter for a subset of fish in parallel
double calculateBarycenter(fish *fishes, int num_fishes, int start, int end) {
    double numerator = 0;
    double denominator = 0;
#pragma omp for
    for (int i = start; i < end; ++i) {
        denominator += calculateDistance(fishes[i]);
        numerator += fishes[i].weight * calculateDistance(fishes[i]);
    }
    return numerator / denominator;
}

// Execute one step of the simulation in parallel
void step(fish *fishes, int num_fish) {
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int fishes_per_thread = num_fish / omp_get_max_threads();
        int start = thread_id * fishes_per_thread;
        int end = (thread_id == omp_get_max_threads() - 1) ? num_fish : start + fishes_per_thread;

        swim(fishes, start, end);

// Wait for all threads to finish updating fish positions
#pragma omp barrier

        updateWeight(fishes, num_fish, start, end);

        // Calculate the barycenter for this thread's group of fish
        double barycenter = calculateBarycenter(fishes, num_fish, start, end);
    }
}

MPI_Datatype MPI_FISH;

int main(int argc, char **argv) {

    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Define the custom MPI_FISH datatype
    int lengths[4] = {1, 1, 1, 1};
    MPI_Aint offsets[4];
    offsets[0] = offsetof(fish, x);
    offsets[1] = offsetof(fish, y);
    offsets[2] = offsetof(fish, weight);
    offsets[3] = offsetof(fish, lastDistance);
    MPI_Datatype types[4] = {MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    MPI_Type_create_struct(4, lengths, offsets, types, &MPI_FISH);
    MPI_Type_commit(&MPI_FISH);

    fish *fishes;
    if (rank == 0){
        fishes = (fish *)malloc(NUM_FISH * sizeof(fish));
        // Check if memory allocation was successful
        if (fishes == NULL) {
            perror("Memory allocation failed");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Initialize fish population with random coordinates and weights
        for (int i = 0; i < NUM_FISH; i++) {
            fishes[i].x = (double)(rand() % 201 - 100);
            fishes[i].y = (double)(rand() % 201 - 100);
            fishes[i].weight = FISH_WEIGHT;
            fishes[i].lastDistance = calculateDistance(fishes[i]);
        }

    }

    int base_fish_per_process = NUM_FISH / size;
    int fish_per_process;
    int remainder = NUM_FISH % size;  // The number of fish that cannot be evenly divided

    if (rank < remainder){
        fish_per_process = base_fish_per_process + 1;
    } else{
        fish_per_process = base_fish_per_process;
    }

    fish *worker = (fish *)malloc(fish_per_process * sizeof(fish));
    if (worker == NULL) {
        perror("Memory allocation failed for worker");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int *send_counts = NULL;
    int *displacements = NULL;

    if (rank == 0){
        send_counts = (int *)malloc(size * sizeof(int));
        displacements = (int *)malloc(size * sizeof(int));

        int disp = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = (i < remainder) ? base_fish_per_process + 1 : base_fish_per_process;
            displacements[i] = disp;
            disp += send_counts[i];
        }
    }

    MPI_Scatterv(fishes, send_counts, displacements, MPI_FISH, worker, fish_per_process, MPI_FISH, 0, MPI_COMM_WORLD);

    // Record simulation start time
    double start = omp_get_wtime();

    // Execute the simulation for a set number of rounds
    for (int i = 0; i < ROUND; ++i) {
        step(worker, fish_per_process);
    }

    MPI_Barrier(MPI_COMM_WORLD);  // Ensures all processes finish before gathering
    MPI_Gatherv(worker, fish_per_process, MPI_FISH, fishes, send_counts, displacements, MPI_FISH, 0, MPI_COMM_WORLD);

    free(worker);

    if (rank == 0) {
        free(send_counts);
        free(displacements);

        // Record simulation end time
        double end = omp_get_wtime();

        // Compute and print the execution time
        double time_taken = end - start;
        printf("Execution time: %.5f seconds\n", time_taken);
        free(fishes);
    }

    MPI_Type_free(&MPI_FISH);  // Free the custom MPI datatype
    MPI_Finalize();

    return 0;
}
