#include "leanstream/predictor.h"
// Resident representative-row GEMV + exact keep exclusion + native top-k.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>
#include <sched.h>
#include <omp.h>
#ifdef __aarch64__
#include <arm_neon.h>
#endif

// Optional exact row-lookahead hints, configured only while workers are drained.
static std::atomic<int> row_prefetch{0};

struct Context {
    int rows, stride, k, threads, precision;
    std::vector<float> scores;
    std::vector<int32_t> candidates;
    std::vector<uint8_t> excluded;
    Context(int r,int s,int k_,int t,int p):rows(r),stride(s),k(k_),threads(t),precision(p),
        scores(r),candidates(r),excluded(r) {}
};

static const std::vector<int>& allowed_cpus() {
    static std::vector<int> cpus=[](){
        cpu_set_t mask; CPU_ZERO(&mask);
        if(sched_getaffinity(0,sizeof(mask),&mask)) return std::vector<int>{};
        std::vector<int> result;
        for(int c=0;c<CPU_SETSIZE;++c) if(CPU_ISSET(c,&mask)) result.push_back(c);
        return result;
    }();
    return cpus;
}

static void dot32(const float* w,const float* x,int s,float& g,float& u) {
#ifdef __aarch64__
    float32x4_t ag[4],au[4];
    for(int j=0;j<4;++j) ag[j]=au[j]=vdupq_n_f32(0);
    for(int i=0;i<s;i+=16) for(int j=0;j<4;++j) {
        auto z=vld1q_f32(x+i+4*j);
        ag[j]=vfmaq_f32(ag[j],vld1q_f32(w+i+4*j),z);
        au[j]=vfmaq_f32(au[j],vld1q_f32(w+s+i+4*j),z);
    }
    g=vaddvq_f32(vaddq_f32(vaddq_f32(ag[0],ag[1]),vaddq_f32(ag[2],ag[3])));
    u=vaddvq_f32(vaddq_f32(vaddq_f32(au[0],au[1]),vaddq_f32(au[2],au[3])));
#else
    g=u=0;
    for(int i=0;i<s;++i) {g+=w[i]*x[i];u+=w[s+i]*x[i];}
#endif
}


#ifndef __aarch64__
static float half_float(uint16_t h) {
    // Portable correctness fallback only. ARM path below widens NEON half loads.
    int e=(h>>10)&31, m=h&1023;
    float value=e==0 ? std::ldexp(float(m),-24) :
                e==31 ? (m ? NAN : INFINITY) : std::ldexp(float(1024+m),e-25);
    return (h&32768) ? -value : value;
}
#endif

static void dot16(const uint16_t* w,const uint16_t* x,int s,float& g,float& u) {
#ifdef __aarch64__
    float32x4_t ag[4],au[4];
    for(int j=0;j<4;++j) ag[j]=au[j]=vdupq_n_f32(0);
    for(int i=0;i<s;i+=16) for(int j=0;j<2;++j) {
        auto z=vld1q_f16(reinterpret_cast<const float16_t*>(x+i+8*j));
        auto a=vld1q_f16(reinterpret_cast<const float16_t*>(w+i+8*j));
        auto b=vld1q_f16(reinterpret_cast<const float16_t*>(w+s+i+8*j));
        auto zl=vcvt_f32_f16(vget_low_f16(z)), zh=vcvt_f32_f16(vget_high_f16(z));
        ag[2*j]=vfmaq_f32(ag[2*j],vcvt_f32_f16(vget_low_f16(a)),zl);
        ag[2*j+1]=vfmaq_f32(ag[2*j+1],vcvt_f32_f16(vget_high_f16(a)),zh);
        au[2*j]=vfmaq_f32(au[2*j],vcvt_f32_f16(vget_low_f16(b)),zl);
        au[2*j+1]=vfmaq_f32(au[2*j+1],vcvt_f32_f16(vget_high_f16(b)),zh);
    }
    g=vaddvq_f32(vaddq_f32(vaddq_f32(ag[0],ag[1]),vaddq_f32(ag[2],ag[3])));
    u=vaddvq_f32(vaddq_f32(vaddq_f32(au[0],au[1]),vaddq_f32(au[2],au[3])));
#else
    g=u=0;
    for(int i=0;i<s;++i) {float z=half_float(x[i]);g+=half_float(w[i])*z;u+=half_float(w[s+i])*z;}
#endif
}

static void dot8(const int8_t* w,const int8_t* x,int s,int32_t& g,int32_t& u) {
#ifdef __aarch64__
    int32x4_t ag0=vdupq_n_s32(0),ag1=ag0,au0=ag0,au1=ag0;
    for(int i=0;i<s;i+=32) {
        auto z0=vld1q_s8(x+i),z1=vld1q_s8(x+i+16);
        ag0=vdotq_s32(ag0,vld1q_s8(w+i),z0);ag1=vdotq_s32(ag1,vld1q_s8(w+i+16),z1);
        au0=vdotq_s32(au0,vld1q_s8(w+s+i),z0);au1=vdotq_s32(au1,vld1q_s8(w+s+i+16),z1);
    }
    g=vaddvq_s32(vaddq_s32(ag0,ag1));u=vaddvq_s32(vaddq_s32(au0,au1));
#else
    g=u=0;for(int i=0;i<s;++i) {g+=int(w[i])*int(x[i]);u+=int(w[s+i])*int(x[i]);}
#endif
}

static void dot4(const int8_t* w,const int8_t* x,int s,int32_t& g,int32_t& u) {
#ifdef __aarch64__
    int32x4_t ag0=vdupq_n_s32(0),ag1=ag0,au0=ag0,au1=ag0;
    for(int i=0;i<s;i+=32) {
        auto z0=vld1q_s8(x+i),z1=vld1q_s8(x+i+16);
        auto a=vld1q_s8(w+i/2),b=vld1q_s8(w+s/2+i/2);
        auto al=vshrq_n_s8(vshlq_n_s8(a,4),4),ah=vshrq_n_s8(a,4);
        auto bl=vshrq_n_s8(vshlq_n_s8(b,4),4),bh=vshrq_n_s8(b,4);
        ag0=vdotq_s32(ag0,al,z0);ag1=vdotq_s32(ag1,ah,z1);
        au0=vdotq_s32(au0,bl,z0);au1=vdotq_s32(au1,bh,z1);
    }
    g=vaddvq_s32(vaddq_s32(ag0,ag1));u=vaddvq_s32(vaddq_s32(au0,au1));
#else
    g=u=0;
    for(int i=0;i<s;i+=32) for(int j=0;j<16;++j) {
        unsigned a=uint8_t(w[i/2+j]),b=uint8_t(w[s/2+i/2+j]);
        int al=int(a&15),ah=int(a>>4),bl=int(b&15),bh=int(b>>4);
        if(al>=8)al-=16;if(ah>=8)ah-=16;if(bl>=8)bl-=16;if(bh>=8)bh-=16;
        g+=al*x[i+j]+ah*x[i+j+16];u+=bl*x[i+j]+bh*x[i+j+16];
    }
#endif
}

extern "C" {
int ls_row_prefetch(int distance){if(distance!=0 && distance!=1 && distance!=4)return -1;row_prefetch.store(distance,std::memory_order_relaxed);return 0;}
int ls_available_cores() {return int(allowed_cpus().size());}
void* ls_create(int rows,int stride,int k,int threads,int precision) {
    if(rows<=0 || stride<=0 || stride%32 || k<1 || k>rows || threads<1 ||
       threads>int(allowed_cpus().size()) || (precision!=32 && precision!=16 && precision!=8 && precision!=4)) return nullptr;
    omp_set_dynamic(0); omp_set_num_threads(threads);
    int failed=0;
    #pragma omp parallel num_threads(threads) reduction(|:failed)
    {
        cpu_set_t mask;CPU_ZERO(&mask);
        CPU_SET(allowed_cpus()[omp_get_thread_num()],&mask);
        if(sched_setaffinity(0,sizeof(mask),&mask)) failed=1;
    }
    if(failed) return nullptr;
    return new Context(rows,stride,k,threads,precision);
}
void ls_destroy(void* context) {delete static_cast<Context*>(context);}
int ls_predict(void* context,const void* weights,const float* scales,const void* input,
               float input_scale,const int32_t* keep,int nkeep,int skip,int32_t* output,
               float* gates,float* ups,float* scores) {
    if(!context || !weights || !input || !output) return -1;
    auto& c=*static_cast<Context*>(context);
    if(nkeep<0 || nkeep>c.rows-c.k || (nkeep && !keep)) return -2;
    if((c.precision==8 || c.precision==4) && (!scales || !std::isfinite(input_scale) || input_scale<=0)) return -1;
    std::fill(c.excluded.begin(),c.excluded.end(),0);
    for(int i=0;i<nkeep;++i) {
        if(keep[i]<0 || keep[i]>=c.rows || c.excluded[keep[i]]) return -2;
        c.excluded[keep[i]]=1;
    }
    const int lookahead=c.precision==4?row_prefetch.load(std::memory_order_relaxed):0;
    #pragma omp parallel for schedule(static) num_threads(c.threads)
    for(int r=0;r<c.rows;++r) {
        if(skip && c.excluded[r]) {
            c.scores[r]=-INFINITY;
            if(gates) gates[r]=NAN;
            if(ups) ups[r]=NAN;
            continue;
        }
        if(lookahead && r+lookahead<c.rows){
            const auto* future=static_cast<const int8_t*>(weights)+size_t(r+lookahead)*c.stride;
            // Gate and up are already adjacent in each row: touch each future
            // packed cache line once, without changing a load or arithmetic.
            for(int b=0;b<c.stride;b+=64)__builtin_prefetch(future+b,0,3);
        }
        float g,u;
        if(c.precision==32) {
            dot32(static_cast<const float*>(weights)+size_t(r)*2*c.stride,
                  static_cast<const float*>(input),c.stride,g,u);
        } else if(c.precision==16) {
            dot16(static_cast<const uint16_t*>(weights)+size_t(r)*2*c.stride,
                  static_cast<const uint16_t*>(input),c.stride,g,u);
        } else {
            int32_t gi,ui;
            if(c.precision==8)
                dot8(static_cast<const int8_t*>(weights)+size_t(r)*2*c.stride,
                     static_cast<const int8_t*>(input),c.stride,gi,ui);
            else
                dot4(static_cast<const int8_t*>(weights)+size_t(r)*c.stride,
                     static_cast<const int8_t*>(input),c.stride,gi,ui);
            g=float(gi)*(input_scale*scales[2*r]);u=float(ui)*(input_scale*scales[2*r+1]);
        }
        c.scores[r]=std::fabs((g/(1.f+std::exp(-g)))*u);
        if(gates) gates[r]=g;
        if(ups) ups[r]=u;
    }
    int n=0;
    for(int r=0;r<c.rows;++r) if(!c.excluded[r]) {
        if(!std::isfinite(c.scores[r])) return -3;
        c.candidates[n++]=r;
    }
    auto better=[&](int a,int b) {return c.scores[a]>c.scores[b] ||
                                     (c.scores[a]==c.scores[b] && a<b);};
    if(n>c.k) std::nth_element(c.candidates.begin(),c.candidates.begin()+c.k,c.candidates.begin()+n,better);
    std::sort(c.candidates.begin(),c.candidates.begin()+c.k,better);
    std::copy_n(c.candidates.begin(),c.k,output);
    if(scores) std::copy(c.scores.begin(),c.scores.end(),scores);
    return 0;
}
// Speculative-only subset: row IDs still address the original packed W4 rows.
// The full ls_predict path remains authoritative for final membership.
int ls_predict_subset(void* context,const void* weights,const float* scales,const void* input,
                      float input_scale,const int32_t* ids,int count,int take,int32_t* output){
    if(!context||!weights||!scales||!input||!ids||!output)return -1;
    auto& c=*static_cast<Context*>(context);
    if(c.precision!=4||count<1||count>c.rows||take<1||take>count)return -2;
    if(!std::isfinite(input_scale)||input_scale<=0)return -1;
    std::fill(c.excluded.begin(),c.excluded.end(),0);
    for(int j=0;j<count;j++){
        int r=ids[j];if(r<0||r>=c.rows||c.excluded[r])return -2;
        c.excluded[r]=1;c.candidates[j]=r;
    }
    #pragma omp parallel for schedule(static) num_threads(c.threads)
    for(int j=0;j<count;j++){
        int r=ids[j];int32_t gi,ui;
        dot4(static_cast<const int8_t*>(weights)+size_t(r)*c.stride,
             static_cast<const int8_t*>(input),c.stride,gi,ui);
        float g=float(gi)*(input_scale*scales[2*r]);
        float u=float(ui)*(input_scale*scales[2*r+1]);
        c.scores[r]=std::fabs((g/(1.f+std::exp(-g)))*u);
    }
    for(int j=0;j<count;j++)if(!std::isfinite(c.scores[ids[j]]))return -3;
    auto better=[&](int a,int b){return c.scores[a]>c.scores[b]||
                                     (c.scores[a]==c.scores[b]&&a<b);};
    if(count>take)std::nth_element(c.candidates.begin(),c.candidates.begin()+take,
                                   c.candidates.begin()+count,better);
    std::sort(c.candidates.begin(),c.candidates.begin()+take,better);
    std::copy_n(c.candidates.begin(),take,output);
    return 0;
}
}
