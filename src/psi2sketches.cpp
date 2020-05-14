#include "parsepsi.h"
#include "blaze/Util.h"
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>
#include <getopt.h>
#include "sketch/mh.h"

void usage(const char *s) {
    std::fprintf(stderr, "usage: %s <opts> inputfile blazeout\n-l: Leafcutter mode\n-h: Usage.\n-p: Set sketch output prefix. Default: empty string.\n", *s);
    std::exit(1);
}

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


int main(int argc, char **argv) {
    std::string outprefix;
    blaze::DynamicMatrix<float> mat;
    int c;
    bool leafcutter = false;
    int nhashes = 100;
    uint64_t seed = 0;
    while((c = getopt(argc, argv, "p:H:Slh?")) >= 0) {
        switch(c) {
            case 'l': leafcutter = true; break;
            case 'p': outprefix = optarg; break;
            case 'H': nhashes = std::atoi(optarg); break;
            case 'S': seed = std::strtoull(optarg, nullptr, 10); break;
            case 'h': usage(argv[0]);
        }
    }
    std::cerr << "outprefix: " << outprefix << '\n';
    if(optind + 2 >= argc) {
        usage(argv[0]);
    }
    mat = parsepsi<>(argv[optind]);
    std::fprintf(stderr, "Now sketching %zu rows, %zu cols\n", mat.rows(), mat.columns());
    std::ofstream header(outprefix + ".header");
    header << "#" << mat.rows() << 'x' << mat.columns() << '.' << nhashes << " 16-bit signatures.\n";
    header.flush();
    sketch::mh::ShrivastavaHash<true, std::uint16_t> hasher(mat.columns(), nhashes, seed);
    std::vector<std::vector<std::uint16_t>> results(mat.rows());
    // Don't have to set maximum weight, since normalized is default
    #pragma omp parallel for
    for(size_t i = 0; i < mat.rows(); ++i) {
        std::cerr << "i: " << i << '\n';
        auto hash = hasher.hash(row(mat, i, blaze::unchecked));
        std::string seedstr;
        if(seed) seedstr = std::string(".") + std::to_string(seed);
        std::string opath = outprefix + '.' + std::to_string(i) + '.' + std::to_string(nhashes) + seedstr + ".psi";
        std::FILE *ofp = std::fopen(opath.data(), "rb");
        if(!ofp) throw 1;
        ::write(::fileno(ofp), (const char *)hash.data(), hash.size() * sizeof(hash[0]));
        std::fclose(ofp);
        results[i] = std::move(hash);
    }
    if(mat.rows() < 1000) {
        blaze::SymmetricMatrix<blaze::DynamicMatrix<float>> dm(mat.rows());
        float inv = 1. / nhashes;
        for(size_t i = 0; i < mat.rows(); ++i) {
            auto p1 = results[i].data();
            for(size_t j = i + 1; j < mat.rows(); ++j) {
                auto p2 = results[j].data();
                int shared = 0;
                for(int k = 0; k < nhashes;++k) shared += p1[k] == p2[k];
                dm(i, j) = shared * inv;
            }
        }
        std::cout << dm << '\n';
    }
}
