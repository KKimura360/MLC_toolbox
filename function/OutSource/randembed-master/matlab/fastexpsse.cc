static inline float
fastpow2 (float p)
{
  float offset = (p < 0) ? 1.0f : 0.0f;
  float clipp = (p < -126) ? -126.0f : p;
  int w = (int) clipp;
  float z = clipp - w + offset;
  union { uint32_t i; float f; } v = { (uint32_t) ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

  return v.f;
}

static inline float
fastexp (float p)
{
  return fastpow2 (1.442695040f * p);
}

#define declconst128(a,b) static const __m128 a = {(b), (b), (b), (b)}

static __m128
vfastpow2 (const __m128 p)
{
  declconst128(zero, 0.0f);
  declconst128(one, 1.0f);
  declconst128(minus126, -126.0f);
  declconst128(c_121_2740838, 121.2740575f);
  declconst128(c_27_7280233, 27.7280233f);
  declconst128(c_4_84252568, 4.84252568f);
  declconst128(c_1_49012907, 1.49012907f);
  declconst128(oneshift23, 1 << 23);
  union { __m128i i; __m128 f; } v;
  
  __m128 ltzero = _mm_cmplt_ps (p, zero);
  __m128 offset = _mm_and_ps (ltzero, one);
  __m128 lt126 = _mm_cmplt_ps (p, minus126);
  __m128 clipp = _mm_or_ps (_mm_andnot_ps (lt126, p), _mm_and_ps (lt126, minus126));
  __m128i w = _mm_cvttps_epi32 (clipp);
  __m128 wf =  _mm_cvtepi32_ps (w);
  __m128 z = _mm_add_ps(clipp, _mm_sub_ps(offset, wf)); /* z = clipp - wf + offset */
 
  v.i = _mm_cvttps_epi32 (
          _mm_mul_ps (oneshift23, 
            _mm_add_ps (clipp,
              _mm_add_ps (c_121_2740838, 
                _mm_sub_ps (
                  _mm_div_ps (c_27_7280233, _mm_sub_ps (c_4_84252568, z)),
                  _mm_mul_ps (c_1_49012907, z)
                )
              )
            )
          )
        );            
  
  return v.f;
}

static inline __m128
vfastexp (const __m128 p)
{
  declconst128(c_invlog_2, 1.442695040f);

  return vfastpow2 (_mm_mul_ps (c_invlog_2, p));
}
