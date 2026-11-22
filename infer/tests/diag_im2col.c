/* diag_im2col.c - verify the im2col panel for 1x1 and 3x3 (CPU oracle). */
#include "vc4_int.h"
#include "gpu_gate.h"
#include "util.h"

static int one(vc4_ctx *ctx, int Cin, int H, int W, int KH, int KW, int pad)
{
    const int hp = H + 2*pad, wp = (W + 2*pad + 15) & ~15;
    const int ho = H + 2*pad - KH + 1, wo = W + 2*pad - KW + 1;
    const uint32_t K = (uint32_t)Cin*KH*KW, nk = K;
    const uint32_t M = (uint32_t)ho*wp;
    float *in = malloc((size_t)Cin*H*W*4);
    if (getenv("SEQ")) for (size_t i=0;i<(size_t)Cin*H*W;i++) in[i]=(float)i;
    else for (size_t i=0;i<(size_t)Cin*H*W;i++) in[i]=frand();
    float *pimg = malloc((size_t)Cin*hp*wp*4);
    memset(pimg,0,(size_t)Cin*hp*wp*4);
    for (int c=0;c<Cin;c++) for (int y=0;y<H;y++)
        memcpy(pimg+((size_t)c*hp + (y+pad))*wp + pad,
               in+((size_t)c*H+y)*W, W*4);
    vc4_buf bi, bp;
    vc4_alloc(ctx, (size_t)Cin*H*W*4, &bi);
    vc4_alloc(ctx, (size_t)M*nk*4, &bp);
    vc4_upload(ctx, &bi, in, (size_t)Cin*H*W*4);
    float *pad_img = vc4__scratch_alloc(ctx, (size_t)Cin*hp*wp*4+64, NULL);
    memset(pad_img,0,(size_t)Cin*hp*wp*4+64);
    memcpy(pad_img, pimg, (size_t)Cin*hp*wp*4);
    float *panel = vc4_host_ptr(ctx,&bp);
    memset(panel, 0xCD, (size_t)M*nk*4);
    vc4__l2_flush(ctx);

    char a1[16],a2[16],a3[8],a4[8],name[64];
    char *argv[] = {"im2col","--cin",a1,"--k",a2,"--kh",a3,"--kw",a4,NULL};
    snprintf(a1,16,"%d",Cin); snprintf(a2,16,"%u",K);
    snprintf(a3,8,"%d",KH); snprintf(a4,8,"%d",KW);
    snprintf(name,64,"im2col_%d_%u_%d_%d",Cin,K,KH,KW);
    vc4_kernel *k=NULL;
    vc4_status st = vc4__kernel(ctx,name,argv,4,&k);
    if (st!=VC4_OK){printf("kernel: %s\n",vc4_strerror(st));return 1;}
    const uint32_t tr = M/16;
    int pnq = (tr%4==0)?4:(tr%2==0)?2:1;
    const uint32_t per = tr/(uint32_t)pnq;
    uint32_t pad_bus = ctx->gpu_bus + (uint32_t)((uint8_t*)pad_img - ctx->arena);
    for (int i=0;i<pnq;i++){
        uint32_t *u=vc4__uni(ctx,k,i);
        u[0]=(uint32_t)hp*wp*4; u[1]=(uint32_t)i*per*64; u[2]=pad_bus;
        u[3]=64; u[4]=vc4_bus_addr(ctx,&bp); u[5]=per; u[6]=(uint32_t)i;
        u[7]=1024; u[8]=64u*nk; u[9]=(uint32_t)wp*4;
        u[10]=(uint32_t)i*per*64u*nk;
    }
    uint64_t ns=0; st=vc4__launch(ctx,k,pnq,&ns);
    usleep(20000); vc4__l2_flush(ctx);
    long bad=0;
    for (uint32_t rb=0; rb<M/16; rb++)
      for (uint32_t kk=0; kk<K; kk++) {
        uint32_t tap = kk / Cin, ci = kk % Cin;
        uint32_t ky = tap / KW, kx = tap % KW;
        for (uint32_t r=0;r<16;r++) {
            uint32_t m = rb*16+r;                 /* padded-image pixel */
            uint32_t oy = m/wp, ox = m%wp;
            int y = (int)oy + (int)ky - pad, x = (int)ox + (int)kx - pad;
            float ref = (y<0||y>=H||x<0||x>=W)?0.0f:in[((size_t)ci*H+y)*W+x];
            float got = panel[((size_t)rb*K+kk)*16+r];
            if ((int)(m % wp) >= wo) continue;   /* discarded by the gather */
            if (memcmp(&ref,&got,4)!=0) { if(bad<3) printf("   rb%u k%u r%u got %.6g ref %.6g\n",rb,kk,r,got,ref); bad++; }
        }
      }
    printf("  panel[0..15]:"); for(int q=0;q<16;q++) printf(" %.1f", panel[q]); printf("\n");
    printf("  ref  [0..15]:");
    for(int r=0;r<16;r++){ uint32_t rb=0,kk=0; uint32_t tap=kk/Cin,ci=kk%Cin;
        uint32_t ky=tap/KW,kx=tap%KW; uint32_t m=rb*16+r,oy=m/wp,ox=m%wp;
        int y=(int)oy+(int)ky-pad,x=(int)ox+(int)kx-pad;
        float rf=(y<0||y>=H||x<0||x>=W)?0.0f:in[((size_t)ci*H+y)*W+x]; printf(" %.1f",rf);}
    printf("\n");
    printf("im2col %dx%dx%d k%d pad%d: %s bad=%ld/%u\n", Cin,H,W,KH,pad,
           vc4_strerror(st), bad, M*K);
    free(in); free(pimg);
    return bad?1:0;
}

int main(void)
{
    if (gpu_gate_skip(__FILE__))
        return GPU_SKIP_STATUS;

    int lock=gpu_lock(); if(lock<0)return 2;
    vc4_ctx *ctx=NULL;
    if(vc4_init(&ctx,NULL,0)!=VC4_OK){printf("init fail\n");return 2;}
    int rc=0;
    rc|=one(ctx,16,8,16,1,1,0);
    rc|=one(ctx,16,12,12,3,3,0);
    rc|=one(ctx,16,16,16,3,3,1);
    vc4_shutdown(ctx); gpu_unlock(lock);
    return rc;
}
