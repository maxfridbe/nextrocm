#pragma once
#include "all_profile.h"
#include "ns_internal.hpp"
int ni2_all_profile_invoke(ns_gpu*,ni2_all_profile_record,int(*)(void*),void*);
template<class F> int ni2_all_profile_call(ns_gpu* g,int family,int type,int rows,int cols,int aux,F&& f) {
 auto callback=[](void* p)->int{return (*static_cast<F*>(p))();};
 return ni2_all_profile_invoke(g,{family,type,rows,cols,aux,0,0.0},callback,&f);
}
