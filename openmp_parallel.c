#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <omp.h>

#define FISH_WEIGHT     1.0     // Initial weight of the fish
#define ROUND           5000    // Number of iterations for the fish to swim
#define NUM_SUBGROUPS   16      // Number of parallel threads/subgroups
#define NUM_FISH        5000    // Total number of fish
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
#pragma omp for schedule(static)
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

#pragma omp parallel for reduction(max: max)
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
    double maxObjective = getMaximumObjective(fishes, num_fishes, 0, num_fishes);

#pragma omp for schedule(static)
    for (int i = start; i < end; ++i) {
        double obj = calculateObjective(fishes[i], num_fishes);
        fishes[i].weight += obj / maxObjective;
        if (fishes[i].weight >= MAX_WEIGHT){
            fishes[i].weight = MAX_WEIGHT;
        }
    }
}

// Function to calculate the barycenter for a subset of fish in parallel
double calculateBarycenter(fish *fishes, int num_fishes, int start, int end) {
    double numerator = 0;
    double denominator = 0;
#pragma omp for schedule(static)
    for (int i = start; i < end; ++i) {
        denominator += calculateDistance(fishes[i]);
        numerator += fishes[i].weight * calculateDistance(fishes[i]);
    }
    return numerator / denominator;
}

// Execute one step of the simulation in parallel
void step(fish *fishes, int num_fish) {
#pragma omp parallel num_threads(NUM_SUBGROUPS)
    {
        int thread_id = omp_get_thread_num();
        int fishes_per_thread = num_fish / NUM_SUBGROUPS;
        int start = thread_id * fishes_per_thread;
        int end = (thread_id == NUM_SUBGROUPS - 1) ? num_fish : start + fishes_per_thread;

        swim(fishes, start, end);

// Wait for all threads to finish updating fish positions
#pragma omp barrier

        updateWeight(fishes, num_fish, start, end);

        // Calculate the barycenter for this thread's group of fish
        double barycenter = calculateBarycenter(fishes, num_fish, start, end);
    }
}

int main(int argc, char **argv) {
    fish *fishes = (fish *)malloc(NUM_FISH * sizeof(fish));

    // Check if memory allocation was successful
    if (fishes == NULL) {
        perror("Memory allocation failed");
        return 1;
    }

    // Initialize fish population with random coordinates and weights
    for (int i = 0; i < NUM_FISH; i++) {
        fishes[i].x = (double)(rand() % 201 - 100);
        fishes[i].y = (double)(rand() % 201 - 100);
        fishes[i].weight = FISH_WEIGHT;
        fishes[i].lastDistance = calculateDistance(fishes[i]);
    }

    // Record simulation start time
    double start = omp_get_wtime();

    // Execute the simulation for a set number of rounds
    for (int i = 0; i < ROUND; ++i) {
        step(fishes, NUM_FISH);
    }

    // Record simulation end time
    double end = omp_get_wtime();

    // Compute and print the execution time
    double time_taken = end - start;
    printf("Execution time: %.5f seconds\n", time_taken);

    // Free the allocated memory for the fish population
    free(fishes);

    return 0;
}
