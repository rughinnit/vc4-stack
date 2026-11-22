/* burner.c - a fixed-instruction CPU workload for the co-run test.
 *
 *   burner --threads N --iters K
 *
 * Each thread runs K iterations of a dependent xorshift/rotate/accumulate
 * chain, so the instruction count is fixed by K and does not depend on the
 * clock or on what else is running.  Wall time and process CPU time are
 * printed as one JSON object; the checksum must be identical on every run,
 * which is what makes "the same work" true across the alone and co-run runs.
 *
 * Build: gcc -O2 -pthread -o burner burner.c
 */
#define _GNU_SOURCE
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint64_t g_iters;
static pthread_barrier_t g_barrier;
static uint64_t g_sink[64];

static double now_s(clockid_t id)
{
	struct timespec ts;
	clock_gettime(id, &ts);
	return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *worker(void *arg)
{
	long idx = (long)arg;
	uint64_t x = 0x9e3779b97f4a7c15ULL ^
		     ((uint64_t)(idx + 1) * 0xbf58476d1ce4e5b9ULL);
	uint64_t acc = 0;
	uint64_t i;

	pthread_barrier_wait(&g_barrier);
	for (i = 0; i < g_iters; i++) {
		x ^= x >> 12;
		x ^= x << 25;
		x ^= x >> 27;
		x *= 0x2545f4914f6cdd1dULL;
		acc += x ^ (x >> 32);
		acc = (acc << 7) | (acc >> 57);
	}
	g_sink[idx & 63] = acc;
	return NULL;
}

int main(int argc, char **argv)
{
	int threads = 1, i;
	pthread_t tid[64];
	double wall0, wall1, cpu0, cpu1;
	uint64_t sum = 0;

	g_iters = 200000000ULL;
	for (i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--threads") && i + 1 < argc)
			threads = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--iters") && i + 1 < argc)
			g_iters = strtoull(argv[++i], NULL, 10);
		else {
			fprintf(stderr, "usage: %s [--threads N] [--iters K]\n",
				argv[0]);
			return 2;
		}
	}
	if (threads < 1 || threads > 64 || g_iters == 0) {
		fprintf(stderr, "burner: threads 1..64, iters > 0\n");
		return 2;
	}
	if (pthread_barrier_init(&g_barrier, NULL, (unsigned)threads)) {
		perror("pthread_barrier_init");
		return 2;
	}
	wall0 = now_s(CLOCK_MONOTONIC);
	cpu0 = now_s(CLOCK_PROCESS_CPUTIME_ID);
	for (i = 0; i < threads; i++) {
		if (pthread_create(&tid[i], NULL, worker, (void *)(long)i)) {
			perror("pthread_create");
			return 2;
		}
	}
	for (i = 0; i < threads; i++)
		pthread_join(tid[i], NULL);
	cpu1 = now_s(CLOCK_PROCESS_CPUTIME_ID);
	wall1 = now_s(CLOCK_MONOTONIC);
	for (i = 0; i < 64; i++)
		sum += g_sink[i];
	printf("{\"threads\": %d, \"iters_per_thread\": %llu, "
	       "\"wall_s\": %.6f, \"cpu_s\": %.6f, \"checksum\": \"%016llx\"}\n",
	       threads, (unsigned long long)g_iters, wall1 - wall0,
	       cpu1 - cpu0, (unsigned long long)sum);
	return 0;
}
