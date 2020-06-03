#ifndef REGCMP_H__
#define REGCMP_H__

namespace regmatch {

template<typename IT>
static INLINE auto count_paired_1bits(IT x) {
    static constexpr IT bitmask = static_cast<IT>(0x5555555555555555uLL);
    return sketch::integral::popcount((x >> 1) & x & bitmask);
}

static inline int count(const uint32_t *SK_RESTRICT lhs, const uint32_t *SK_RESTRICT rhs, unsigned num_reg) {
    int ret = 0;
    unsigned i = 0;
#ifdef __AVX512BW__
    const unsigned numvecs = num_reg / 16;
    const __m512i *lhp(reinterpret_cast<const __m512i *>(lhs));
    const __m512i *rhp(reinterpret_cast<const __m512i *>(rhs));

    if((uint64_t)lhs % sizeof(__m512i) || (uint64_t)rhs % sizeof(__m512i)) {
        #pragma GCC unroll 4
        for(;i < numvecs; ++i) {
            ret += sketch::popcount(_mm512_cmpeq_epi32_mask(_mm512_loadu_si512(lhp + i), _mm512_loadu_si512(rhp + i)));
        }
    } else {
        #pragma GCC unroll 4
        for(;i < numvecs; ++i) {
            ret += sketch::popcount(_mm512_cmpeq_epi32_mask(_mm512_load_si512(lhp + i), _mm512_load_si512(rhp + i)));
        }
    }
    i *= 16;
#elif __AVX2__
    const unsigned nsimd = (num_reg / 8) * 8;
    const __m256i *vp1 = (const __m256i *)(lhs);
    const __m256i *vp2 = (const __m256i *)(rhs);
    for(;i < nsimd; i += 8)
        ret += sketch::popcount(_mm256_movemask_ps(reinterpret_cast<__m256>(_mm256_cmpeq_epi32(_mm256_loadu_si256(vp1++), _mm256_loadu_si256(vp2++)))));
#elif __SSE2__
    const unsigned nsimd = (num_reg / 4) * 4;
    const __m128i *vp1 = (const __m128i *)(lhs);
    const __m128i *vp2 = (const __m128i *)(rhs);
    for(;i < nsimd; i += 4)
        ret += sketch::popcount(_mm_movemask_ps(reinterpret_cast<__m128>(_mm_cmpeq_epi32(_mm_loadu_si128(vp1++), _mm_loadu_si128(vp2++)))));
#endif
    for(;i < num_reg;++i) ret += lhs[i] == rhs[i];
    return ret;
}
static inline int count(const uint8_t *SK_RESTRICT lhs, const uint8_t *SK_RESTRICT rhs, unsigned num_reg) {
    int ret = 0;
    unsigned i = 0;
#ifdef __AVX512BW__
    const unsigned numvecs = num_reg / 64;
    const __m512i *lhp(reinterpret_cast<const __m512i *>(lhs));
    const __m512i *rhp(reinterpret_cast<const __m512i *>(rhs));

    if((uint64_t)lhs % sizeof(__m512i) || (uint64_t)rhs % sizeof(__m512i)) {
        #pragma GCC unroll 4
        for(;i < numvecs; ++i) {
            ret += sketch::popcount(_mm512_cmpeq_epi8_mask(_mm512_loadu_si512(lhp + i), _mm512_loadu_si512(rhp + i)));
        }
    } else {
        #pragma GCC unroll 4
        for(;i < numvecs; ++i) {
            ret += sketch::popcount(_mm512_cmpeq_epi8_mask(_mm512_load_si512(lhp + i), _mm512_load_si512(rhp + i)));
        }
    }
    i *= 64;
#elif __AVX2__
    const unsigned nsimd = (num_reg / 32) * 32;
    const __m256i *vp1 = (const __m256i *)(lhs);
    const __m256i *vp2 = (const __m256i *)(rhs);
    for(;i < nsimd; i += 32)
        ret += sketch::popcount(_mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_loadu_si256(vp1++), _mm256_loadu_si256(vp2++))));
#elif __SSE2__
    const unsigned nsimd = (num_reg / 16) * 16;
    const __m128i *vp1 = (const __m128i *)(lhs);
    const __m128i *vp2 = (const __m128i *)(rhs);
    for(;i < nsimd; i += 16)
        ret += sketch::popcount(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(vp1++), _mm_loadu_si128(vp2++))));
#endif
    for(;i < num_reg;++i) ret += lhs[i] == rhs[i];
    return ret;
}
static inline int count(const uint16_t *SK_RESTRICT lhs, const uint16_t *SK_RESTRICT rhs, unsigned num_reg) {
    int ret = 0;
    unsigned i = 0;
#ifdef __AVX2__
    const unsigned nsimd = (num_reg / 16) * 16;
    const __m256i *vp1 = (const __m256i *)(lhs);
    const __m256i *vp2 = (const __m256i *)(rhs);
    for(;i < nsimd; i += 16)
        ret += count_paired_1bits(_mm256_movemask_epi8(_mm256_cmpeq_epi16(_mm256_loadu_si256(vp1++), _mm256_loadu_si256(vp2++))));
    num_reg -= i;
    lhs = (const uint16_t *)vp1;
    rhs = (const uint16_t *)vp2;
#elif __SSE2__
    const unsigned nsimd = (num_reg / 8) * 8;

    const __m128i *vp1 = (const __m128i *)(lhs);
    const __m128i *vp2 = (const __m128i *)(rhs);
    for(;i < nsimd; i += 8)
        ret += count_paired_1bits(_mm_movemask_epi8(_mm_cmpeq_epi16(_mm_loadu_si128(vp1++), _mm_loadu_si128(vp2++))));
    num_reg -= i;
    lhs = (const uint16_t *)vp1;
    rhs = (const uint16_t *)vp2;
#endif
    while(num_reg >= 8) {
        ret += *lhs == *rhs;
        ret += lhs[1] == rhs[1];
        ret += lhs[2] == rhs[2];
        ret += lhs[3] == rhs[3];
        ret += lhs[4] == rhs[4];
        ret += lhs[5] == rhs[5];
        ret += lhs[6] == rhs[6];
        ret += lhs[7] == rhs[7];
        lhs += 8;
        rhs += 8;
        num_reg -= 8;
    }
    while(num_reg--) ret += *lhs++ == *rhs++;
    return ret;
}

} // namespace regmatch


#endif /* REGCMP_H__*/
