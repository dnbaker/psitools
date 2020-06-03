#include "parsepsi.h"
#include "blaze/Util.h"
#include <fstream>
#include <map>
#include <iostream>
#include <string>
#include <unistd.h>
#include <getopt.h>
#include "sketch/mh.h"
#include "timer.h"
#include "mergepsis.h"
#ifdef _OPENMP
#include <omp.h>
#endif

template<typename IT>
static INLINE auto count_paired_1bits(IT x) {
    static constexpr IT bitmask = static_cast<IT>(0x5555555555555555uLL);
    return sketch::integral::popcount((x >> 1) & x & bitmask);
}

void usage(const char *s) {
    std::fprintf(stderr, "usage: %s <opts> inputfile blazeout\n-l: Leafcutter mode\n-h: Usage.\n-p: Set sketch output prefix. Default: empty string.\n"
                         "-m: multifile mode. Treat inputfile as a list of files, one per line, merge them into one experiment, and then sketch.\n"
                         "-M: Use the maximum observed PSI as a cap\n"
                         "-H: number of hashes [100]\n", s);
    std::exit(1);
}
auto gett() {return std::chrono::high_resolution_clock::now();}
using util::timediff2ms;

#ifndef REGISTERTYPE
#define REGISTERTYPE uint16_t
#endif

using RegisterType = REGISTERTYPE;

auto &update(std::map<RegisterType, uint32_t> &out, const std::map<RegisterType, uint32_t> &in) {
    for(const auto &pair: in)
        out[pair.first] += pair.second;
    return out;
}

void print_args(char **argv) {
    std::fprintf(stderr, "Command-line: ");
    do std::fprintf(stderr, "%s ", *argv); while(*++argv);
    std::fputc('\n', stderr);
}


static inline int matching_registers(const uint32_t *lhs, const uint32_t *rhs, unsigned num_reg) {
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

static inline int matching_registers(const uint8_t *lhs, const uint8_t *rhs, unsigned num_reg) {
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
static inline int matching_registers(const uint16_t *lhs, const uint16_t *rhs, unsigned num_reg) {
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


int main(int argc, char **argv) {
    print_args(argv);
    std::string outprefix;
    int c;
    bool leafcutter = false;
    bool multifile = false;
    bool skip_matrix = false;
    bool normalize = true;
    unsigned nhashes = 100;
    uint64_t seed = 0;
    while((c = getopt(argc, argv, "t:p:H:S:NsmMlh?")) >= 0) {
        switch(c) {
            case 's': skip_matrix = true; break;
            case 'l': leafcutter = true; break;
            case 'p': outprefix = optarg; break;
            case 't': omp_set_num_threads(std::atoi(optarg)); break;
            case 'H': nhashes = std::atoi(optarg); break;
            case 'N': normalize = false; break;
            case 'S': seed = std::strtoull(optarg, nullptr, 10); break;
            case 'm': multifile = true; break;
            case 'h': usage(argv[0]);
        }
    }
    std::cerr << "outprefix: " << outprefix << '\n';
    if(optind + 1 != argc) {
        usage(argv[0]);
    }
    std::cerr << "Getting mat\n";
    blaze::DynamicMatrix<float> mat;
    if(multifile) {
        std::vector<std::string> paths;
        std::ifstream ifs(argv[optind]);
        for(std::string line;std::getline(ifs, line);) {
            paths.emplace_back(std::move(line));
        }
        mat = files2master(paths, leafcutter, normalize);
    } else {
        mat = leafcutter ? leafcutter_parsepsi<>(argv[optind], normalize): parsepsi<>(argv[optind], normalize);
    }
    if(!normalize) {
        outprefix = outprefix + ".unnorm";
    } else {
        mat = blaze::clamp(mat, 0.f, 1.f);
    }
    {
        blaze::Archive<std::ofstream> arch("save.features.blaze");
        arch << mat;
        std::FILE *ofp = std::fopen("save.features.raw", "wb");
        for(size_t i = 0; i < mat.rows(); ++i)
            std::fwrite(&mat(i, 0), mat.columns(), sizeof(float), ofp);
        std::fclose(ofp);
    }
    std::fprintf(stderr, "Now sketching %zu rows, %zu cols\n", mat.rows(), mat.columns());
    std::ofstream header(outprefix + ".header");
    header << "#" << mat.rows() << 'x' << mat.columns() << ". " << nhashes << " 16-bit signatures.\n";
    header.flush();
    sketch::mh::ShrivastavaHash<true, RegisterType> hasher(mat.columns(), nhashes, seed);
#if PERFORM_REDUCTION
    std::map<RegisterType, uint32_t> counts;
#endif
    std::vector<std::vector<RegisterType>> results(mat.rows());
    auto start = gett();
    if(!normalize) {
        hasher.set_threshold(blaze::max(mat));
    }
    // Don't have to set maximum weight, since normalized is default
#if PERFORM_REDUCTION
    #pragma omp declare reduction (merge : std::map<RegisterType, uint32_t> : update(omp_out, omp_in))

    #pragma omp parallel for reduction(merge: counts)
#else
    #pragma omp parallel for
#endif
    for(size_t i = 0; i < mat.rows(); ++i) {
        auto hash = hasher.hash(row(mat, i, blaze::unchecked));
        std::string seedstr;
        if(seed) seedstr = std::string(".") + std::to_string(seed);
        std::string opath = outprefix + '.' + std::to_string(i) + '.' + std::to_string(nhashes) + seedstr + ".psi";
        std::FILE *ofp = std::fopen(opath.data(), "wb");
        if(!ofp) throw 1;
        ::write(::fileno(ofp), (const char *)hash.data(), hash.size() * sizeof(hash[0]));
        std::fclose(ofp);
        results[i] = std::move(hash);
#if PERFORM_REDUCTION
        for(const auto v: results[i])
            ++counts[v];
#endif
    }
#if PERFORM_REDUCTION
    size_t totalsum = 0;
    for(const auto &pair: counts) {
        std::fprintf(stderr, "value %u has occured %u times\n", pair.first, pair.second);
        totalsum += pair.second * pair.first;
    }
    std::fprintf(stderr, "total trials %zu for %zu hashes for an average of %0.12g\n", totalsum, mat.rows() * nhashes, double(totalsum) / (mat.rows() * nhashes));
    assert(std::accumulate(counts.begin(), counts.end(), size_t(0), [](auto x, auto y) {return x + y.second;}) == mat.rows() * nhashes);
#endif
    auto stop = gett();
    auto t = timediff2ms(start, stop);
    std::fprintf(stderr, "Sketching took %gms\n", t);
    if(!skip_matrix) {
        start = gett();
        blaze::SymmetricMatrix<blaze::DynamicMatrix<float>> dm(mat.rows());
        const float inv = 1. / nhashes;
        OMP_PFOR
        for(size_t i = 0; i < mat.rows(); ++i) {
            dm(i,i) = 1.;
            auto p1 = results[i].data();
            for(size_t j = i + 1; j < mat.rows(); ++j) {
                dm(i, j) = matching_registers(p1, results[j].data(), nhashes) * inv;
            }
        }
        stop = gett();
        std::fprintf(stderr, "Comparing sketches took %gms\n", timediff2ms(start, stop));
        blaze::Archive<std::ofstream> arch("dist.blaze");
        arch << dm;
        std::FILE *tmpfp = std::fopen("dist.float32.np", "w");
        if(!tmpfp) throw 2;
        const int fd = ::fileno(tmpfp);
        for(size_t i = 0; i < dm.rows(); ++i)
            ::write(fd, row(dm, i, blaze::unchecked).data(), dm.columns() * sizeof(float));
        std::fclose(tmpfp);
    }
}
