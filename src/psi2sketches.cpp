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

void usage(const char *s) {
    std::fprintf(stderr, "usage: %s <opts> inputfile blazeout\n-l: Leafcutter mode\n-h: Usage.\n-p: Set sketch output prefix. Default: empty string.\n"
                         "-m: multifile mode. Treat inputfile as a list of files, one per line, merge them into one experiment, and then sketch.\n"
                         "-M: Use the maximum observed PSi as a cap\n"
                         "-H: number of hashes [100]\n", s);
    std::exit(1);
}
auto gett() {return std::chrono::high_resolution_clock::now();}
using util::timediff2ms;

#if 0
template<bool weighted, typename Signature=std::uint16_t, typename IndexType=std::uint32_t,
         typename FT=float>
struct ShrivastavaHash {
    const IndexType nd_;
    const IndexType nh_;
    const uint64_t seedseed_;
    FT mv_;
    std::unique_ptr<FT[]> maxvals_;
    schism::Schismatic<IndexType> div_;
public:
#endif

auto &update(std::map<uint16_t, uint32_t> &out, const std::map<uint16_t, uint32_t> &in) {
    for(const auto &pair: in)
        out[pair.first] += pair.second;
    return out;
}


int main(int argc, char **argv) {
    std::string outprefix;
    int c;
    bool leafcutter = false;
    bool use_max_as_cap = false;
    bool multifile = false;
    int nhashes = 100;
    uint64_t seed = 0;
    while((c = getopt(argc, argv, "p:H:S:mMlh?")) >= 0) {
        switch(c) {
            case 'l': leafcutter = true; break;
            case 'p': outprefix = optarg; break;
            case 'H': nhashes = std::atoi(optarg); break;
            case 'S': seed = std::strtoull(optarg, nullptr, 10); break;
            case 'M': use_max_as_cap = true; break;
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
        mat = files2master(paths, leafcutter);
    } else {
        mat = leafcutter ? leafcutter_parsepsi<>(argv[optind]): parsepsi<>(argv[optind]);
    }
    mat = blaze::clamp(mat, 0.f, 1.f);
    std::fprintf(stderr, "Now sketching %zu rows, %zu cols\n", mat.rows(), mat.columns());
    std::ofstream header(outprefix + ".header");
    header << "#" << mat.rows() << 'x' << mat.columns() << '.' << nhashes << " 16-bit signatures.\n";
    header.flush();
    sketch::mh::ShrivastavaHash<true, std::uint16_t> hasher(mat.columns(), nhashes, seed);
    std::map<uint16_t, uint32_t> counts;
    std::vector<std::vector<std::uint16_t>> results(mat.rows());
    auto start = gett();
    if(use_max_as_cap) {
        blaze::DynamicVector<float, blaze::rowVector> maxv = blaze::max<blaze::columnwise>(mat);
        hasher.set_threshold(maxv.data());
    }
    // Don't have to set maximum weight, since normalized is default
    #pragma omp declare reduction (merge : std::map<uint16_t, uint32_t> : update(omp_out, omp_in))

    #pragma omp parallel for reduction(merge: counts)
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
        for(const auto v: results[i])
            ++counts[v];
    }
    for(const auto &pair: counts) {
        std::fprintf(stderr, "value %u has occured %u times\n", pair.first, pair.second);
    }
    assert(std::accumulate(counts.begin(), counts.end(), size_t(0), [](auto x, auto y) {return x + y.second;}) == mat.rows() * nhashes);
    auto stop = gett();
    auto t = timediff2ms(start, stop);
    std::fprintf(stderr, "Sketching took %gms\n", t);
    if(mat.rows() < 1000) {
        start = gett();
        blaze::SymmetricMatrix<blaze::DynamicMatrix<float>> dm(mat.rows());
        float inv = 1. / nhashes;
        OMP_PFOR
        for(size_t i = 0; i < mat.rows(); ++i) {
            dm(i,i) = 1.;
            auto p1 = results[i].data();
            for(size_t j = i + 1; j < mat.rows(); ++j) {
                auto p2 = results[j].data();
                int shared = 0;
                for(int k = 0; k < nhashes;++k) shared += p1[k] == p2[k];
                dm(i, j) = shared * inv;
            }
        }
        stop = gett();
        t = timediff2ms(start, stop);
        std::fprintf(stderr, "Comparing sketches took %gms\n", t);
        std::cout << dm << '\n';
        start = gett();
        OMP_PFOR
        for(size_t i = 0; i < mat.rows(); ++i) {
            dm(i, i) = 0.;
            auto r = row(mat, i, blaze::unchecked);
            auto dmr = row(dm, i, blaze::unchecked);
            for(size_t j = i + 1; j < mat.rows(); ++j)
                dmr[j] = blaze::l2Norm(r - row(mat, j, blaze::unchecked));
        }
        stop = gett();
        t = timediff2ms(start, stop);
        std::fprintf(stderr, "Comparing full vectors took %gms\n", t);
        std::cout << dm << '\n';
    }
}
