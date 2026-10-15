#include "leanstream/predictor.h"
#include <cassert>
#include <cstdio>
int main(){
    constexpr int rows=4,stride=32,k=2;
    float weights[rows*2*stride],input[stride];int32_t ids[k];
    for(auto& x:input)x=1;
    for(int r=0;r<rows;r++)for(int i=0;i<2*stride;i++)weights[r*2*stride+i]=float(r+1)/32;
    void* p=ls_create(rows,stride,k,1,32);assert(p);
    assert(ls_predict(p,weights,nullptr,input,1,nullptr,0,0,ids,nullptr,nullptr,nullptr)==0);
    assert(ids[0]==3&&ids[1]==2);
    std::printf("ranked logical groups: %d %d\n",ids[0],ids[1]);ls_destroy(p);
}
