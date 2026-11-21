#include "../sd15.h"
#include <stddef.h>
_Static_assert(sizeof(nrsd_weight)==16,"weight ABI");
_Static_assert(offsetof(nrsd_weight,dtype)==8,"weight dtype offset");
_Static_assert(sizeof(nrsd_linear_shape)==40,"linear ABI");
_Static_assert(sizeof(nrsd_conv_shape)==104,"conv ABI");
_Static_assert(sizeof(nrsd_attention_shape)==144,"attention ABI");
_Static_assert(offsetof(nrsd_attention_shape,scale)==136,"attention scale offset");
_Static_assert(offsetof(nrsd_attention_shape,causal)==140,"attention causal offset");
int main(void){return 0;}
