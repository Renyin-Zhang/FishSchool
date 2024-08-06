#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <ctype.h>
#include <omp.h>
#include <stddef.h>
#include <mpi.h>

#define FISH_WEIGHT     1.0     // Initial weight of the fish
#define ROUND           5000    // Number of iterations for the fish to swim
#define NUM_FISH        5000    // Total number of fish
#define MAX_WEIGHT      2*FISH_WEIGHT


MPI_Datatype MPI_FISH;


typedef struct {
    double x;                   // x-coordinate of fish
    double y;                   // y-coordinate of fish
    double weight;              // weight of fish
    double lastDistance;        // last computed distance from origin
} fish;

int main(int argc, char **argv) {


    int rank, size;
    FILE *fp1, *fp2, *fopen();
    int process_id, number_of_processes;

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
    if (rank == 0) {
        fp1 = fopen("out1.txt", "w+");
        if (fp1 == NULL) {
            perror("Error opening the files");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fishes = (fish *) malloc(NUM_FISH * sizeof(fish));
        if (fishes == NULL) {
            perror("Memory allocation failed");
            return 1;
        }

        // Initialize fish population with random coordinates and weights
        for (int i = 0; i < NUM_FISH; i++) {
            fishes[i].x = (double) (rand() % 201 - 100);
            fishes[i].y = (double) (rand() % 201 - 100);
            fishes[i].weight = FISH_WEIGHT;
        }
        for (int i = 0; i < NUM_FISH; ++i) {
            fprintf(fp1, "%f %f %f\n", fishes[i].x, fishes[i].y, fishes[i].weight);
        }
        fclose(fp1);
    }

    // Handling the remainder
    int base_fish_per_process = NUM_FISH / size;
    int fish_per_process;
    int remainder = NUM_FISH % size;

    if (rank < remainder) {
        fish_per_process = base_fish_per_process + 1;
    } else {
        fish_per_process = base_fish_per_process;
    }

    fish *worker = (fish *) malloc(fish_per_process * sizeof(fish));
    int *send_counts = NULL;
    int *displacements = NULL;

    if (rank == 0) {
        send_counts = (int *) malloc(size * sizeof(int));
        displacements = (int *) malloc(size * sizeof(int));
        for (int i = 0, disp = 0; i < size; i++) {
            send_counts[i] = (i < remainder) ? base_fish_per_process + 1 : base_fish_per_process;
            displacements[i] = disp;
            disp += send_counts[i];
        }
    }

    MPI_Scatterv(fishes, send_counts, displacements, MPI_FISH, worker, fish_per_process, MPI_FISH, 0, MPI_COMM_WORLD);
    MPI_Gatherv(worker, fish_per_process, MPI_FISH, fishes, send_counts, displacements, MPI_FISH, 0, MPI_COMM_WORLD);

    free(worker);

    if (rank == 0) {
        fp2 = fopen("out2.txt", "w+");
        for (int i = 0; i < NUM_FISH; ++i) {
            fprintf(fp2, "%f %f %f\n", fishes[i].x, fishes[i].y, fishes[i].weight);
        }
        fclose(fp2);
        free(send_counts);
        free(displacements);
    }

    MPI_Finalize();

    return 0;
}
