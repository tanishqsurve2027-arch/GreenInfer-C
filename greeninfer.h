/*
 * ============================================================================
 *  GreenInfer-C : Multiobjective Genetic Optimization of Accuracy, Latency,
 *                 Energy and Memory for C-Based Edge-AI Inference
 * ============================================================================
 *  File   : greeninfer.h
 *  Purpose: All shared constants, data structures and function prototypes.
 *
 *  Think of this file as the "table of contents" of the project:
 *    - Section A : the problem we are modelling (the neural network)
 *    - Section B : the hardware we pretend to run on (an edge microcontroller)
 *    - Section C : the search space (which knobs the GA is allowed to turn)
 *    - Section D : Genetic Algorithm settings
 *    - Section E : Fitness-function weights
 *    - Section F : Data structures (structs)
 *    - Section G : Function prototypes (grouped by team member)
 * ============================================================================
 */

#ifndef GREENINFER_H
#define GREENINFER_H

/* ---------------------------------------------------------------------------
 * Section A : The AI model we are optimizing
 * ---------------------------------------------------------------------------
 * A small fully-connected neural network (MLP) that classifies 28x28
 * grayscale images (e.g. handwritten digits) into 10 classes.
 *   input layer (784) -> [hidden layers] -> output layer (10)
 */
#define INPUT_SIZE   784     /* 28 x 28 pixels                       */
#define OUTPUT_SIZE  10      /* 10 output classes (digits 0..9)      */

/* ---------------------------------------------------------------------------
 * Section B : Simulated edge hardware (a Cortex-M7 class microcontroller)
 * ---------------------------------------------------------------------------
 * We do NOT need a real board: these numbers let us *estimate* cost.
 */
#define CPU_FREQ_MHZ        216.0  /* CPU clock speed                         */
#define LAYER_OVERHEAD_MS   0.05   /* fixed cost to launch each layer (ms)    */
#define STATIC_POWER_MW     5.0    /* power drawn just by being switched on   */
#define FLASH_PJ_PER_BYTE   15.0   /* energy to read 1 byte of weights (pJ)   */
#define SPARSE_EFFICIENCY   0.80   /* sparse kernels skip 80% of pruned work  */

/* ---------------------------------------------------------------------------
 * Section C : Search space (the "genes" of each model)
 * ---------------------------------------------------------------------------
 * Allowed bit-widths and layer widths are listed in main.c as arrays.
 */
#define MIN_LAYERS      1    /* fewest hidden layers allowed              */
#define MAX_LAYERS      6    /* most hidden layers allowed                */
#define MAX_SPARSITY    90   /* max pruning: 90% of weights set to zero   */
#define SPARSITY_STEP   5    /* sparsity moves in steps of 5%             */

/* ---------------------------------------------------------------------------
 * Section D : Genetic Algorithm settings
 * ---------------------------------------------------------------------------
 */
#define POP_SIZE         30    /* number of models in each generation      */
#define NUM_GENERATIONS  30    /* how many generations we evolve           */
#define TOURNAMENT_SIZE  3     /* models competing in each tournament      */
#define CROSSOVER_RATE   0.90  /* chance two parents mix their genes       */
#define MUTATION_RATE    0.20  /* chance EACH gene gets randomly nudged    */
#define ELITE_COUNT      2     /* best models copied unchanged each gen    */

/* ---------------------------------------------------------------------------
 * Section E : Fitness function (weighted multi-objective score)
 * ---------------------------------------------------------------------------
 * Weights must add up to 1.0. They express what we care about most.
 */
#define W_ACCURACY   0.40
#define W_LATENCY    0.20
#define W_ENERGY     0.25
#define W_MEMORY     0.15

#define ACC_TARGET        0.96   /* minimum acceptable accuracy (96%)          */
#define PENALTY_PER_POINT 0.50   /* fitness lost per 1% accuracy below target  */
#define ACC_SCALE_LOW     0.70   /* accuracy mapped to score 0.0               */
#define ACC_SCALE_HIGH    0.99   /* accuracy mapped to score 1.0               */

#define MAX_ARCHIVE  4096        /* storage for every unique model ever seen   */

/* ---------------------------------------------------------------------------
 * Section F : Data structures
 * ---------------------------------------------------------------------------
 */

/* The 4 objectives (plus parameter counts) measured for one model. */
typedef struct {
    double accuracy;        /* 0.0 - 1.0  (higher is better)              */
    double latency_ms;      /* time for one inference (lower is better)   */
    double energy_mj;       /* energy for one inference (lower is better) */
    double memory_kb;       /* flash + RAM footprint (lower is better)    */
    long   params_total;    /* weights before pruning                     */
    long   params_nonzero;  /* weights that survive pruning               */
} Metrics;

/* One candidate Edge-AI model = one "individual" in the GA population.
 * The first four fields are its GENES (what the GA changes). */
typedef struct {
    int     bit_width;      /* GENE 1: 4, 8, 16 or 32 bits per weight     */
    int     sparsity_pct;   /* GENE 2: % of weights pruned (0..90)        */
    int     num_layers;     /* GENE 3: number of hidden layers (1..6)     */
    int     layer_width;    /* GENE 4: neurons per hidden layer           */
    Metrics m;              /* measured objectives                        */
    double  fitness;        /* single combined score (higher is better)   */
} ModelCandidate;

/* ---------------------------------------------------------------------------
 * Section G : Function prototypes (grouped by presenter)
 * ---------------------------------------------------------------------------
 */

/* Member 1 - Model representation & objective estimation */
ModelCandidate make_baseline(void);
void   random_candidate(ModelCandidate *c);
long   count_parameters(const ModelCandidate *c);
double estimate_accuracy(const ModelCandidate *c);
double estimate_latency(const ModelCandidate *c);
double estimate_memory(const ModelCandidate *c);
double estimate_energy(const ModelCandidate *c);
void   compute_metrics(ModelCandidate *c);

/* Member 2 - Fitness & Pareto analysis */
void compute_fitness(ModelCandidate *c, const Metrics *ref);
int  dominates(const ModelCandidate *a, const ModelCandidate *b);
void archive_add(const ModelCandidate *c);
void print_pareto_front(void);

/* Member 3 - Genetic Algorithm operators */
void init_population(ModelCandidate pop[], int n, const Metrics *ref);
const ModelCandidate *tournament_select(const ModelCandidate pop[], int n);
void crossover(const ModelCandidate *p1, const ModelCandidate *p2,
               ModelCandidate *child);
void mutate(ModelCandidate *c);

/* Member 4 - Terminal UI / reporting */
void print_banner(unsigned int seed);
void print_candidate_details(const char *title, const ModelCandidate *c);
void print_table_header(void);
void print_generation_row(int gen, const ModelCandidate *best, double avg_fit);
void print_comparison(const ModelCandidate *base, const ModelCandidate *opt);

#endif /* GREENINFER_H */
