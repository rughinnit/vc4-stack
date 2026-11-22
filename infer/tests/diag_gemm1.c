/* diag_gemm1.c - bmode=1 (packed B) GEMM with CPU-built panels. */
#include "vc4_int.h"
#include "gpu_gate.h"
#include "util.h"

int main(int argc, char **argv)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int lock = gpu_lock(); if (lock < 0) return 2;
    vc4_ctx *ctx = NULL;
    if (vc4_init(&ctx, NULL, 0) != VC4_OK) { printf("init fail\n"); return 2; }
    int M = argc>3?atoi(argv[1]):32, N = argc>3?atoi(argv[2]):16, K = argc>3?atoi(argv[3]):16, rc = 0;
    float *A = malloc(M*K*4), *B = malloc(K*N*4), *Apan = malloc(M*K*4), *Bpan = malloc(K*N*4);
    double *R = malloc(M*N*sizeof(double));
    for (int i=0;i<M*K;i++) A[i]=frand();
    for (int i=0;i<K*N;i++) B[i]=frand();
    ref_gemm(A, B, R, M, N, K);
    /* A panel: PANEL[(rb*K + k)*16 + r] = A[rb*16+r][k] */
    for (int rb=0; rb<M/16; rb++)
      for (int k=0;k<K;k++)
        for (int r=0;r<16;r++)
          Apan[(rb*K+k)*16+r] = A[(rb*16+r)*K+k];
    /* B panel: [cb][k][co] = B[k][cb*16+co] */
    for (int cb=0; cb<N/16; cb++)
      for (int k=0;k<K;k++)
        for (int co=0;co<16;co++)
          Bpan[(cb*K+k)*16+co] = B[(size_t)k*N + cb*16+co];

    vc4_buf pa,pb,pc;
    vc4_alloc(ctx, M*K*4, &pa); vc4_alloc(ctx, K*N*4, &pb); vc4_alloc(ctx, M*N*4, &pc);
    vc4_upload(ctx, &pa, Apan, M*K*4);
    vc4_upload(ctx, &pb, Bpan, K*N*4);
    vc4_status st = vc4__gemm_run(ctx, M, N, K, 1, vc4_bus_addr(ctx,&pb),
                                  vc4_bus_addr(ctx,&pc), vc4_bus_addr(ctx,&pa));
    float *C = malloc(M*N*4);
    vc4_download(ctx, &pc, C, M*N*4);
    long mm=0; double worst=0;
    for (int i=0;i<M*N;i++){ double e=fabs(C[i]-R[i]); if(e>worst)worst=e;
        if(e>1e-4*(fabs(R[i])>1?fabs(R[i]):1)){ if(mm<4) printf("  mm[%d] got %.6g ref %.6g\n",i,C[i],R[i]); mm++; } }
    printf("bmode=1 M=%d N=%d K=%d: %s mismatches=%ld/%d worst=%.3g\n",
           M,N,K, vc4_strerror(st), mm, M*N, worst);
    rc = mm?1:0;
    vc4_shutdown(ctx); gpu_unlock(lock);
    return rc;
}
