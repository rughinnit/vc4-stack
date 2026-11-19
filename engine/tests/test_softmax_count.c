/* controlled correct-count test */
#include "vc4infer.h"
#include "util.h"
#include <math.h>
int main(void){
    int lock=gpu_lock(); if(lock<0) return 2;
    vc4_ctx *ctx=NULL; if(vc4_init(&ctx,NULL,0)!=VC4_OK) return 2;
    enum { R=8, C=16 };
    float z[R*C]; int32_t lab[R]; int amv[R];
    for(int r=0;r<R;r++){
        int am=0; float mx=-1e30f;
        for(int c=0;c<C;c++){ z[r*C+c]=frand()*3.0f; if(z[r*C+c]>mx){mx=z[r*C+c];am=c;} }
        amv[r]=am;
        lab[r]= (r&1) ? am : (am+1)%C;
    }
    uint32_t groups=R/8;
    vc4_buf bz,bl,bd,bp;
    vc4_alloc(ctx,sizeof(z)+64,&bz); vc4_alloc(ctx,sizeof(lab)+64,&bl);
    vc4_alloc(ctx,sizeof(z)+64,&bd); vc4_alloc(ctx,groups*64,&bp);
    vc4_upload(ctx,&bz,z,sizeof(z)); vc4_upload(ctx,&bl,lab,sizeof(lab));
    float *zero=calloc(groups*16,4); vc4_upload(ctx,&bp,zero,groups*64); free(zero);
    /* probe: store the per-row correct flag instead of dlogits */
    setenv("VC4_SOFTMAX_PROBE","cor",1);
    vc4_status st=vc4_softmax_ce(ctx,&bz,&bl,&bd,&bp,R,1.0f/R);
    if(st!=VC4_OK){printf("FAIL: %s\n",vc4_strerror(st));return 1;}
    float *fl=calloc(R*C,4); vc4_download(ctx,&bd,fl,R*64);
    for(int r=0;r<R;r++){
        printf("row %d: am=%2d lab=%2d ref_flag=%d got_flag=%.0f zy=%.6f max=%.6f\n",
          r,amv[r],lab[r],(amv[r]==lab[r]),fl[r*C],z[r*C+lab[r]],z[r*C+amv[r]]);
    }
    unsetenv("VC4_SOFTMAX_PROBE");
    vc4_shutdown(ctx); gpu_unlock(lock); return 0;
}
