#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <omp.h>

#define FISH_WEIGHT 1.0         // Initial weight of the fish
#define ROUND       1000        // Number of iterations for the fish to swim
#define NUM_FISH    1000        // Total number of fish
#define MAX_WEIGHT  2*FISH_WEIGHT

// Structure definition for a fish
typedef struct {
    double x;                // x-coordinate of fish
    double y;                // y-coordinate of fish
    double weight;           // weight of fish
    double lastDistance;     // last computed distance from origin
}fish;

// Function to calculate distance of a fish from the origin
double calculateDistance(fish fish){
    double x = fish.x;
    double y = fish.y;
    return sqrt(x*x + y*y);
}

// Function to calculate the objective value for a fish
double calculateObjective(fish fish, int num_fish) {
    return (calculateDistance(fish) - fish.lastDistance);
}

// Simulate the swimming behavior of the fish
void swim(fish *fishes, int num_fish) {
    for (int i = 0; i < num_fish; ++i) {
        double swimX = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
        double swimY = ((double)rand() / RAND_MAX) * 0.2 - 0.1;
        fishes[i].lastDistance = calculateDistance(fishes[i]); // Store the last distance

        fishes[i].x += swimX;  // Update x-coordinate
        fishes[i].y += swimY;  // Update y-coordinate
    }
}

// Function to get the maximum objective among all fish
double getMaximumObjective(fish *fishes, int num_fishes){
    double max = 0;
    for (int i = 0; i < num_fishes; ++i) {
        double obj = calculateObjective(fishes[i], num_fishes);
        if (obj >= max) {
            max = obj;
        }
    }
    return max;
}

// Update the weights of the fish based on their objective values
void updateWeight(fish *fishes, int num_fishes){
    double maxObjective = getMaximumObjective(fishes, num_fishes);
    for (int i = 0; i < num_fishes; ++i) {
        double obj = calculateObjective(fishes[i], num_fishes);
        fishes[i].weight += obj / maxObjective;  // Update weight relative to max objective
        if (fishes[i].weight >= MAX_WEIGHT){
            fishes[i].weight = MAX_WEIGHT;
        }
    }
}

// Function to calculate the barycenter (or center of mass) for the fish population
double calculateBarycenter(fish *fishes, int num_fishes) {
    double numerator = 0;
    double denominator = 0;
    for (int i = 0; i < num_fishes; ++i) {
        denominator += calculateDistance(fishes[i]);
    }
    for (int i = 0; i < num_fishes; ++i) {
        numerator += fishes[i].weight * calculateDistance(fishes[i]);
    }
    return numerator/denominator;
}

// Execute one step of the simulation
void step(fish *fishes, int fish_num){
    swim(fishes, fish_num);
    updateWeight(fishes, fish_num);
    calculateBarycenter(fishes, fish_num);
}

int main(int argc, char ** argv) {

    // Allocate memory for fish population
    fish* fishes = (fish*)malloc(NUM_FISH * sizeof(fish));

    // Check if memory allocation was successful
    if (fishes == NULL) {
        perror("Memory allocation failed");
        return 1;
    }

    // Initialize fish population with random coordinates and initial weights
    for (int i = 0; i < NUM_FISH; i++) {
        fishes[i].x = (double)(rand() % 201 - 100);
        fishes[i].y = (double)(rand() % 201 - 100);
        fishes[i].weight = FISH_WEIGHT;
        fishes[i].lastDistance = calculateDistance(fishes[i]);
    }

    // Record simulation start time
    double start = omp_get_wtime();
    int repetitions = 20000;
    // Execute the simulation for a set number of rounds
    for (int j = 0; j < repetitions; ++j) {
        for (int i = 0; i < ROUND; ++i) {
            step(fishes, NUM_FISH);
        }
    }

    // Record simulation end time
    double end = omp_get_wtime();

    // Compute and print the time taken for the simulation
    double time_taken = end - start;
    printf("Execution time: %.5f seconds\n", time_taken/repetitions);

    // Clean up dynamically allocated memory
    free(fishes);

    return 0;
}
