/*
 * ============================================================================
 *  GreenInfer-C : Multiobjective Genetic Optimization of Accuracy, Latency,
 *                 Energy and Memory for C-Based Edge-AI Inference
 * ============================================================================
 *  File   : main.c
 *  Build  : gcc -std=c99 -Wall -Wextra -O2 main.c -o greeninfer -lm
 *  Run    : ./greeninfer          (fixed seed 42 -> same result every time)
 *           ./greeninfer 7       (different seed -> different search path)
 *           ./greeninfer 0       (seed taken from the current time)
 *
 *  Layout of this file (one PART per team member):
 *    PART 0 : Small helpers (random numbers, clamping, labels)
 *    PART 1 : Model representation & the 4 objective estimators
 *    PART 2 : Fitness function, Pareto dominance, Pareto archive
 *    PART 3 : Genetic Algorithm operators (init, select, crossover, mutate)
 *    PART 4 : Terminal UI (tables and final comparison)
 *    PART 5 : main() - the generation loop that ties everything together
 * ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "greeninfer.h"

/* Allowed values for the two "list" genes. The GA only ever picks from these. */
static const int BIT_OPTIONS[]   = {4, 8, 16, 32};
static const int WIDTH_OPTIONS[] = {32, 64, 128, 256, 512, 1024};

#define NUM_BIT_OPTIONS   ((int)(sizeof(BIT_OPTIONS) / sizeof(BIT_OPTIONS[0])))
#define NUM_WIDTH_OPTIONS ((int)(sizeof(WIDTH_OPTIONS) / sizeof(WIDTH_OPTIONS[0])))

/* ============================================================================
 *  PART 0 : Small helpers
 * ============================================================================
 */

/* Random real number in [0, 1). */
static double rand_unit(void)
{
    return rand() / (RAND_MAX + 1.0);
}

/* Random integer in [lo, hi] (both ends included). */
static int rand_int(int lo, int hi)
{
    return lo + (int)(rand_unit() * (hi - lo + 1));
}

/* Keep a value inside [0, 1]. */
static double clamp01(double x)
{
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

/* Keep an integer inside [lo, hi]. */
static int clamp_int(int x, int lo, int hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Position of 'value' inside 'arr' (or 0 if not found). */
static int index_of(const int arr[], int n, int value)
{
    int i;
    for (i = 0; i < n; i++) {
        if (arr[i] == value) return i;
    }
    return 0;
}

/* Human-friendly name for a bit-width, e.g. 8 -> "INT8". */
static const char *precision_label(int bits)
{
    switch (bits) {
        case 32: return "FP32";
        case 16: return "FP16";
        case 8:  return "INT8";
        default: return "INT4";
    }
}

/* Short text describing a model's genes, e.g. "INT8 s30% 4x256". */
static void format_config(const ModelCandidate *c, char *buf, size_t size)
{
    snprintf(buf, size, "%s s%d%% %dx%d", precision_label(c->bit_width),
             c->sparsity_pct, c->num_layers, c->layer_width);
}

/* ============================================================================
 *  PART 1 : Model representation & objective estimation   (Member 1)
 * ============================================================================
 *  None of these functions run a real neural network. They are ANALYTICAL
 *  COST MODELS: simple formulas that behave the way real hardware does
 *  (fewer bits -> less memory and energy, more pruning -> less work, etc.).
 */

/* The "initial model" a developer would naively deploy:
 * full 32-bit floats, no pruning, 4 hidden layers of 512 neurons. */
ModelCandidate make_baseline(void)
{
    ModelCandidate c;
    memset(&c, 0, sizeof(c));
    c.bit_width    = 32;
    c.sparsity_pct = 0;
    c.num_layers   = 4;
    c.layer_width  = 512;
    return c;
}

/* Fill a candidate with random genes (used to create generation 1). */
void random_candidate(ModelCandidate *c)
{
    memset(c, 0, sizeof(*c));
    c->bit_width    = BIT_OPTIONS[rand_int(0, NUM_BIT_OPTIONS - 1)];
    c->sparsity_pct = rand_int(0, MAX_SPARSITY / SPARSITY_STEP) * SPARSITY_STEP;
    c->num_layers   = rand_int(MIN_LAYERS, MAX_LAYERS);
    c->layer_width  = WIDTH_OPTIONS[rand_int(0, NUM_WIDTH_OPTIONS - 1)];
}

/* Number of weights + biases in the MLP (before pruning).
 *   input->hidden1 : INPUT_SIZE * width
 *   hidden->hidden : (layers - 1) * width * width
 *   hiddenN->output: width * OUTPUT_SIZE
 *   biases         : layers * width + OUTPUT_SIZE                       */
long count_parameters(const ModelCandidate *c)
{
    long w = c->layer_width;
    long L = c->num_layers;
    return INPUT_SIZE * w + (L - 1) * w * w + w * OUTPUT_SIZE
         + L * w + OUTPUT_SIZE;
}

/* Number of multiply-accumulate (MAC) operations actually executed.
 * Pruned weights are skipped, but sparse kernels are not perfect, so only
 * SPARSE_EFFICIENCY (80%) of the pruned work is really saved. */
static double effective_macs(const ModelCandidate *c)
{
    double sparsity = c->sparsity_pct / 100.0;
    return (double)count_parameters(c) * (1.0 - SPARSE_EFFICIENCY * sparsity);
}

/* Bytes needed to store the weights in flash memory. */
static double weight_storage_bytes(const ModelCandidate *c)
{
    long   total    = count_parameters(c);
    double nonzero  = total * (1.0 - c->sparsity_pct / 100.0);
    double weights  = nonzero * c->bit_width / 8.0;
    /* A pruned model also needs a bitmask: 1 bit per original weight that
     * says "this weight is kept / removed". */
    double mask     = (c->sparsity_pct > 0) ? total / 8.0 : 0.0;
    return weights + mask;
}

/* (a) ACCURACY - a synthetic but realistic validation score.
 *  1. Bigger networks have more capacity -> higher accuracy (saturating).
 *  2. A few layers help; too many layers slightly hurt (harder to train).
 *  3. Fewer bits = rounding error = small accuracy loss.
 *  4. Pruning is almost free at first but hurts a lot near 90%.          */
double estimate_accuracy(const ModelCandidate *c)
{
    double params   = (double)count_parameters(c);
    double sparsity = c->sparsity_pct / 100.0;
    double acc;

    /* 1. capacity: grows from 68% towards 95% as parameters increase */
    acc = 0.68 + 0.27 * (1.0 - exp(-params / 120000.0));

    /* 2. depth bonus up to 4 layers, small penalty after that */
    if (c->num_layers <= 4)
        acc += 0.008 * c->num_layers;
    else
        acc += 0.032 - 0.006 * (c->num_layers - 4);

    /* 3. quantization loss */
    switch (c->bit_width) {
        case 16: acc -= 0.001; break;
        case 8:  acc -= 0.005; break;
        case 4:  acc -= 0.025; break;
        default: break;                 /* FP32: no loss */
    }

    /* 4. pruning loss grows with the 4th power of sparsity */
    acc -= 0.25 * pow(sparsity, 4.0);

    if (acc < 0.0)   acc = 0.0;
    if (acc > 0.999) acc = 0.999;
    return acc;
}

/* (b) LATENCY in milliseconds for ONE inference.
 *  time = (MACs x cycles-per-MAC) / clock-speed  + per-layer overhead
 *  Lower precision lets the CPU's SIMD unit do several MACs per cycle. */
double estimate_latency(const ModelCandidate *c)
{
    double cycles_per_mac;
    double compute_ms;

    switch (c->bit_width) {
        case 32: cycles_per_mac = 4.00; break;  /* float multiply-add      */
        case 16: cycles_per_mac = 2.00; break;
        case 8:  cycles_per_mac = 1.00; break;  /* SIMD int8               */
        default: cycles_per_mac = 0.75; break;  /* int4 needs unpacking    */
    }

    /* cycles / (MHz * 1000) = milliseconds */
    compute_ms = effective_macs(c) * cycles_per_mac / (CPU_FREQ_MHZ * 1000.0);
    return compute_ms + LAYER_OVERHEAD_MS * (c->num_layers + 1);
}

/* (d) MEMORY footprint in KB = weights in flash + activation buffers in RAM.
 *  Two activation buffers are needed (input and output of a layer).
 *  Activations are never stored in less than 8 bits. */
double estimate_memory(const ModelCandidate *c)
{
    int    act_bits  = (c->bit_width < 8) ? 8 : c->bit_width;
    int    widest    = (c->layer_width > INPUT_SIZE) ? c->layer_width : INPUT_SIZE;
    double act_bytes = 2.0 * widest * act_bits / 8.0;

    return (weight_storage_bytes(c) + act_bytes) / 1024.0;
}

/* (c) ENERGY in millijoules for ONE inference, from three sources:
 *    compute : every MAC costs some picojoules (fewer bits = cheaper)
 *    memory  : every weight byte read from flash costs energy
 *    static  : the chip leaks power for as long as inference runs
 *  Must be called AFTER latency is known (static energy depends on it). */
double estimate_energy(const ModelCandidate *c)
{
    double pj_per_mac;
    double compute_pj, memory_pj, static_pj;

    switch (c->bit_width) {
        case 32: pj_per_mac = 450.0; break;
        case 16: pj_per_mac = 220.0; break;
        case 8:  pj_per_mac =  90.0; break;
        default: pj_per_mac =  70.0; break;
    }

    compute_pj = effective_macs(c) * pj_per_mac;
    memory_pj  = weight_storage_bytes(c) * FLASH_PJ_PER_BYTE;
    /* mW x ms = microjoules = 1e6 picojoules */
    static_pj  = STATIC_POWER_MW * c->m.latency_ms * 1e6;

    return (compute_pj + memory_pj + static_pj) * 1e-9;   /* pJ -> mJ */
}

/* Run all four estimators and store the results inside the candidate. */
void compute_metrics(ModelCandidate *c)
{
    c->m.params_total   = count_parameters(c);
    c->m.params_nonzero = (long)(c->m.params_total * (1.0 - c->sparsity_pct / 100.0));
    c->m.accuracy       = estimate_accuracy(c);
    c->m.latency_ms     = estimate_latency(c);
    c->m.memory_kb      = estimate_memory(c);
    c->m.energy_mj      = estimate_energy(c);     /* needs latency first */
}

/* ============================================================================
 *  PART 2 : Fitness function & Pareto analysis   (Member 2)
 * ============================================================================
 */

/* Weighted multi-objective fitness. Every objective is turned into a
 * score between 0 and 1, then the scores are mixed using the weights.
 *
 *   accuracy score = where accuracy falls between 70% (0.0) and 99% (1.0)
 *   latency  score = fraction of latency SAVED vs. the baseline model
 *   energy   score = fraction of energy  SAVED vs. the baseline model
 *   memory   score = fraction of memory  SAVED vs. the baseline model
 *
 *   fitness = 0.40*acc + 0.20*lat + 0.25*energy + 0.15*memory
 *
 * A model below the accuracy target (96%) loses 0.5 fitness for every
 * percentage point it is short, so the GA cannot "cheat" by shrinking the
 * model until it is useless. */
void compute_fitness(ModelCandidate *c, const Metrics *ref)
{
    double acc_score = clamp01((c->m.accuracy - ACC_SCALE_LOW) /
                               (ACC_SCALE_HIGH - ACC_SCALE_LOW));
    double lat_score = clamp01(1.0 - c->m.latency_ms / ref->latency_ms);
    double eng_score = clamp01(1.0 - c->m.energy_mj  / ref->energy_mj);
    double mem_score = clamp01(1.0 - c->m.memory_kb  / ref->memory_kb);

    double fitness = W_ACCURACY * acc_score
                   + W_LATENCY  * lat_score
                   + W_ENERGY   * eng_score
                   + W_MEMORY   * mem_score;

    if (c->m.accuracy < ACC_TARGET)
        fitness -= PENALTY_PER_POINT * (ACC_TARGET - c->m.accuracy) * 100.0;

    c->fitness = fitness;
}

/* Pareto dominance: model A "dominates" model B if A is at least as good
 * as B in ALL four objectives and strictly better in at least one.
 * Returns 1 if a dominates b, otherwise 0. */
int dominates(const ModelCandidate *a, const ModelCandidate *b)
{
    int no_worse = a->m.accuracy   >= b->m.accuracy   &&
                   a->m.latency_ms <= b->m.latency_ms &&
                   a->m.energy_mj  <= b->m.energy_mj  &&
                   a->m.memory_kb  <= b->m.memory_kb;

    int better   = a->m.accuracy   >  b->m.accuracy   ||
                   a->m.latency_ms <  b->m.latency_ms ||
                   a->m.energy_mj  <  b->m.energy_mj  ||
                   a->m.memory_kb  <  b->m.memory_kb;

    return no_worse && better;
}

/* Archive = list of every UNIQUE model evaluated during the whole run.
 * Used at the end to find the Pareto front (all best trade-offs). */
static ModelCandidate archive[MAX_ARCHIVE];
static int archive_count = 0;

static int same_genes(const ModelCandidate *a, const ModelCandidate *b)
{
    return a->bit_width    == b->bit_width    &&
           a->sparsity_pct == b->sparsity_pct &&
           a->num_layers   == b->num_layers   &&
           a->layer_width  == b->layer_width;
}

void archive_add(const ModelCandidate *c)
{
    int i;
    for (i = 0; i < archive_count; i++) {
        if (same_genes(&archive[i], c)) return;   /* already stored */
    }
    if (archive_count < MAX_ARCHIVE)
        archive[archive_count++] = *c;
}

/* qsort helper: lowest energy first. */
static int compare_energy(const void *a, const void *b)
{
    double ea = ((const ModelCandidate *)a)->m.energy_mj;
    double eb = ((const ModelCandidate *)b)->m.energy_mj;
    return (ea > eb) - (ea < eb);
}

/* Print every model that meets the accuracy target and is NOT dominated
 * by any other model that meets the target. These are the "best possible
 * trade-offs": to improve one objective you must sacrifice another. */
void print_pareto_front(void)
{
    static ModelCandidate front[MAX_ARCHIVE];
    int front_count = 0;
    int i, j;
    char cfg[32];

    for (i = 0; i < archive_count; i++) {
        int dominated = 0;
        if (archive[i].m.accuracy < ACC_TARGET) continue;
        for (j = 0; j < archive_count && !dominated; j++) {
            if (j != i && archive[j].m.accuracy >= ACC_TARGET &&
                dominates(&archive[j], &archive[i]))
                dominated = 1;
        }
        if (!dominated) front[front_count++] = archive[i];
    }

    qsort(front, front_count, sizeof(ModelCandidate), compare_energy);

    printf("\n  PARETO FRONT (non-dominated models with accuracy >= %.0f%%)\n",
           ACC_TARGET * 100.0);
    printf("  Explored %d unique models; %d are Pareto-optimal. "
           "Showing up to 10:\n", archive_count, front_count);
    printf("  +----+---------------------+---------+-------------+------------+------------+\n");
    printf("  |  # | Config              | Acc (%%) | Latency(ms) | Energy(mJ) | Memory(KB) |\n");
    printf("  +----+---------------------+---------+-------------+------------+------------+\n");
    for (i = 0; i < front_count && i < 10; i++) {
        format_config(&front[i], cfg, sizeof(cfg));
        printf("  | %2d | %-19s | %7.2f | %11.3f | %10.4f | %10.1f |\n",
               i + 1, cfg, front[i].m.accuracy * 100.0, front[i].m.latency_ms,
               front[i].m.energy_mj, front[i].m.memory_kb);
    }
    printf("  +----+---------------------+---------+-------------+------------+------------+\n");
}

/* ============================================================================
 *  PART 3 : Genetic Algorithm operators   (Member 3)
 * ============================================================================
 */

/* Create 'n' random models and evaluate each one. */
void init_population(ModelCandidate pop[], int n, const Metrics *ref)
{
    int i;
    for (i = 0; i < n; i++) {
        random_candidate(&pop[i]);
        compute_metrics(&pop[i]);
        compute_fitness(&pop[i], ref);
    }
}

/* TOURNAMENT SELECTION: pick TOURNAMENT_SIZE random models and return the
 * fittest. Good models win more often, but weaker ones still get a chance,
 * which keeps the population diverse. */
const ModelCandidate *tournament_select(const ModelCandidate pop[], int n)
{
    const ModelCandidate *best = &pop[rand_int(0, n - 1)];
    int i;
    for (i = 1; i < TOURNAMENT_SIZE; i++) {
        const ModelCandidate *rival = &pop[rand_int(0, n - 1)];
        if (rival->fitness > best->fitness) best = rival;
    }
    return best;
}

/* UNIFORM CROSSOVER: for each gene, flip a coin to decide which parent the
 * child inherits it from. With probability (1 - CROSSOVER_RATE) the child
 * is simply a copy of parent 1. */
void crossover(const ModelCandidate *p1, const ModelCandidate *p2,
               ModelCandidate *child)
{
    *child = *p1;
    if (rand_unit() >= CROSSOVER_RATE) return;

    child->bit_width    = (rand_unit() < 0.5) ? p1->bit_width    : p2->bit_width;
    child->sparsity_pct = (rand_unit() < 0.5) ? p1->sparsity_pct : p2->sparsity_pct;
    child->num_layers   = (rand_unit() < 0.5) ? p1->num_layers   : p2->num_layers;
    child->layer_width  = (rand_unit() < 0.5) ? p1->layer_width  : p2->layer_width;
}

/* MUTATION: each gene has a MUTATION_RATE chance of a SMALL random step,
 * e.g. INT8 -> INT4 or FP16, sparsity 30% -> 40%, 4 layers -> 3 layers.
 * Small steps let the GA fine-tune good models instead of jumping randomly. */
void mutate(ModelCandidate *c)
{
    int step = (rand_unit() < 0.5) ? -1 : 1;      /* direction of change */
    int idx;

    if (rand_unit() < MUTATION_RATE) {             /* gene 1: bit-width */
        idx = index_of(BIT_OPTIONS, NUM_BIT_OPTIONS, c->bit_width);
        idx = clamp_int(idx + step, 0, NUM_BIT_OPTIONS - 1);
        c->bit_width = BIT_OPTIONS[idx];
    }
    if (rand_unit() < MUTATION_RATE) {             /* gene 2: sparsity  */
        c->sparsity_pct += step * rand_int(1, 2) * SPARSITY_STEP;
        c->sparsity_pct  = clamp_int(c->sparsity_pct, 0, MAX_SPARSITY);
    }
    if (rand_unit() < MUTATION_RATE) {             /* gene 3: depth     */
        c->num_layers = clamp_int(c->num_layers + step, MIN_LAYERS, MAX_LAYERS);
    }
    if (rand_unit() < MUTATION_RATE) {             /* gene 4: width     */
        idx = index_of(WIDTH_OPTIONS, NUM_WIDTH_OPTIONS, c->layer_width);
        idx = clamp_int(idx + step, 0, NUM_WIDTH_OPTIONS - 1);
        c->layer_width = WIDTH_OPTIONS[idx];
    }
}

/* qsort helper: highest fitness first. */
static int compare_fitness_desc(const void *a, const void *b)
{
    double fa = ((const ModelCandidate *)a)->fitness;
    double fb = ((const ModelCandidate *)b)->fitness;
    return (fa < fb) - (fa > fb);
}

/* ============================================================================
 *  PART 4 : Terminal UI   (Member 4)
 * ============================================================================
 */

void print_banner(unsigned int seed)
{
    printf("\n");
    printf("==========================================================================================\n");
    printf("   GreenInfer-C : Multiobjective Genetic Optimization for C-Based Edge-AI Inference\n");
    printf("   Objectives  : maximise Accuracy | minimise Latency, Energy, Memory\n");
    printf("==========================================================================================\n");
    printf("   Population: %d   Generations: %d   Tournament: %d   Crossover: %.0f%%   Mutation: %.0f%%\n",
           POP_SIZE, NUM_GENERATIONS, TOURNAMENT_SIZE,
           CROSSOVER_RATE * 100.0, MUTATION_RATE * 100.0);
    printf("   Weights   : Acc %.2f  Lat %.2f  Energy %.2f  Mem %.2f   Accuracy target: %.0f%%   Seed: %u\n",
           W_ACCURACY, W_LATENCY, W_ENERGY, W_MEMORY, ACC_TARGET * 100.0, seed);
    printf("==========================================================================================\n");
}

void print_candidate_details(const char *title, const ModelCandidate *c)
{
    printf("\n  %s\n", title);
    printf("    Precision : %s (%d-bit)    Pruning: %d%%    Hidden layers: %d x %d neurons\n",
           precision_label(c->bit_width), c->bit_width, c->sparsity_pct,
           c->num_layers, c->layer_width);
    printf("    Params    : %ld total, %ld non-zero\n",
           c->m.params_total, c->m.params_nonzero);
    printf("    Accuracy  : %.2f%%   Latency: %.3f ms   Energy: %.4f mJ   Memory: %.1f KB\n",
           c->m.accuracy * 100.0, c->m.latency_ms, c->m.energy_mj, c->m.memory_kb);
}

void print_table_header(void)
{
    printf("\n  EVOLUTION LOG (best model of each generation)\n");
    printf("  +-----+---------------------+---------+-------------+------------+------------+---------+---------+\n");
    printf("  | Gen | Best Config         | Acc (%%) | Latency(ms) | Energy(mJ) | Memory(KB) | Fitness | Avg Fit |\n");
    printf("  +-----+---------------------+---------+-------------+------------+------------+---------+---------+\n");
}

void print_generation_row(int gen, const ModelCandidate *best, double avg_fit)
{
    char cfg[32];
    format_config(best, cfg, sizeof(cfg));
    printf("  | %3d | %-19s | %7.2f | %11.3f | %10.4f | %10.1f | %7.4f | %7.3f |\n",
           gen, cfg, best->m.accuracy * 100.0, best->m.latency_ms,
           best->m.energy_mj, best->m.memory_kb, best->fitness, avg_fit);
}

/* Percentage saved going from 'before' to 'after' (positive = saving). */
static double pct_saved(double before, double after)
{
    return (before - after) / before * 100.0;
}

void print_comparison(const ModelCandidate *base, const ModelCandidate *opt)
{
    char cfg_b[32], cfg_o[32];
    format_config(base, cfg_b, sizeof(cfg_b));
    format_config(opt,  cfg_o, sizeof(cfg_o));

    printf("\n  FINAL COMPARISON : Initial Model vs GreenInfer-C Optimized Model\n");
    printf("  +-------------------+---------------------+---------------------+------------------+\n");
    printf("  | Metric            | Initial (baseline)  | GreenInfer-C        | Change           |\n");
    printf("  +-------------------+---------------------+---------------------+------------------+\n");
    printf("  | Configuration     | %-19s | %-19s |                  |\n", cfg_b, cfg_o);
    printf("  | Accuracy (%%)      | %19.2f | %19.2f | %+7.2f pts      |\n",
           base->m.accuracy * 100.0, opt->m.accuracy * 100.0,
           (opt->m.accuracy - base->m.accuracy) * 100.0);
    printf("  | Latency (ms)      | %19.3f | %19.3f | %6.1f%% faster   |\n",
           base->m.latency_ms, opt->m.latency_ms,
           pct_saved(base->m.latency_ms, opt->m.latency_ms));
    printf("  | Energy (mJ)       | %19.4f | %19.4f | %6.1f%% saved  <<|\n",
           base->m.energy_mj, opt->m.energy_mj,
           pct_saved(base->m.energy_mj, opt->m.energy_mj));
    printf("  | Memory (KB)       | %19.1f | %19.1f | %6.1f%% saved  <<|\n",
           base->m.memory_kb, opt->m.memory_kb,
           pct_saved(base->m.memory_kb, opt->m.memory_kb));
    printf("  | Fitness           | %19.4f | %19.4f |                  |\n",
           base->fitness, opt->fitness);
    printf("  +-------------------+---------------------+---------------------+------------------+\n");

    printf("\n  ******************************************************************************\n");
    printf("  *  GREEN SAVINGS:  Energy -%.1f%%   |   Memory -%.1f%%   |   %.1fx faster\n",
           pct_saved(base->m.energy_mj, opt->m.energy_mj),
           pct_saved(base->m.memory_kb, opt->m.memory_kb),
           base->m.latency_ms / opt->m.latency_ms);
    printf("  *  Accuracy cost:  %.2f percentage points (target was >= %.0f%%)\n",
           (base->m.accuracy - opt->m.accuracy) * 100.0, ACC_TARGET * 100.0);
    printf("  *  Fits in 1 MB microcontroller flash?  Baseline: %s   Optimized: %s\n",
           base->m.memory_kb <= 1024.0 ? "YES" : "NO ",
           opt->m.memory_kb  <= 1024.0 ? "YES" : "NO ");
    printf("  ******************************************************************************\n");
}

/* ============================================================================
 *  PART 5 : main() - the Genetic Algorithm generation loop
 * ============================================================================
 */
int main(int argc, char *argv[])
{
    static ModelCandidate population[POP_SIZE];
    static ModelCandidate next_gen[POP_SIZE];
    ModelCandidate baseline, best_ever;
    unsigned int seed = 42;           /* fixed seed = repeatable demo */
    clock_t start;
    int gen, i;

    /* Optional seed from the command line (0 = use current time). */
    if (argc > 1) {
        seed = (unsigned int)strtoul(argv[1], NULL, 10);
        if (seed == 0) seed = (unsigned int)time(NULL);
    }
    srand(seed);
    start = clock();

    /* STEP 1: Evaluate the unoptimized baseline. It is the reference point
     *         for every "% saved" score in the fitness function. */
    baseline = make_baseline();
    compute_metrics(&baseline);
    compute_fitness(&baseline, &baseline.m);

    print_banner(seed);
    print_candidate_details("INITIAL MODEL (what a developer would deploy without optimization)",
                            &baseline);

    /* STEP 2: Generation 1 = random models. */
    init_population(population, POP_SIZE, &baseline.m);
    best_ever = population[0];

    print_table_header();

    for (gen = 1; gen <= NUM_GENERATIONS; gen++) {
        double total_fit = 0.0;

        /* STEP 3: Rank the population, best first. */
        qsort(population, POP_SIZE, sizeof(ModelCandidate), compare_fitness_desc);

        /* STEP 4: Record statistics and remember the best model ever seen. */
        for (i = 0; i < POP_SIZE; i++) {
            total_fit += population[i].fitness;
            archive_add(&population[i]);
        }
        if (population[0].fitness > best_ever.fitness)
            best_ever = population[0];

        print_generation_row(gen, &population[0], total_fit / POP_SIZE);

        if (gen == NUM_GENERATIONS) break;     /* no need to breed after last gen */

        /* STEP 5: Elitism - the top ELITE_COUNT models survive unchanged. */
        for (i = 0; i < ELITE_COUNT; i++)
            next_gen[i] = population[i];

        /* STEP 6: Fill the rest of the next generation with children:
         *         select two parents -> crossover -> mutate -> evaluate. */
        for (i = ELITE_COUNT; i < POP_SIZE; i++) {
            const ModelCandidate *p1 = tournament_select(population, POP_SIZE);
            const ModelCandidate *p2 = tournament_select(population, POP_SIZE);
            ModelCandidate child;

            crossover(p1, p2, &child);
            mutate(&child);
            compute_metrics(&child);
            compute_fitness(&child, &baseline.m);
            next_gen[i] = child;
        }

        /* STEP 7: The children become the new population. */
        memcpy(population, next_gen, sizeof(population));
    }
    printf("  +-----+---------------------+---------+-------------+------------+------------+---------+---------+\n");

    /* STEP 8: Report the results. */
    print_candidate_details("GREENINFER-C OPTIMIZED MODEL (best found)", &best_ever);
    print_comparison(&baseline, &best_ever);
    print_pareto_front();

    printf("\n  GA finished in %.1f ms of CPU time.\n\n",
           (double)(clock() - start) * 1000.0 / CLOCKS_PER_SEC);
    return 0;
}
