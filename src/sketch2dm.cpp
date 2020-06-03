#include "sketch/sketch.h"
#include "regcmp.h"
#include <fstream>
#include <sys/stat.h>

#ifndef TYPE
#define TYPE uint16_t
using Type = TYPE;
#endif

size_t getfsz(const std::string &path) {
    struct stat buf;
    ::stat(path.data(), &buf);
    return buf.st_size;
}

int main(int argc, char **argv) {
    std::vector<std::string> paths;
    if(argc == 1) throw 1;
    for(char **p = argv + 1; *p; ++p) {
        paths.emplace_back(*p);
    }
    if(paths.size() == 1) {
        std::vector<std::string> tmp;
        std::ifstream ifs(paths.front());
        for(std::string l;std::getline(ifs, l);) {
            tmp.emplace_back(std::move(l));
        }
        std::swap(tmp, paths);
    }
    std::vector<std::vector<Type>> sketches(paths.size());
    std::fprintf(stderr, "Loading %zu sketches\n", sketches.size());
    OMP_PFOR
    for(size_t i = 0; i < paths.size(); ++i) {
        auto &s(sketches[i]);
        s.resize(getfsz(paths[i]) / sizeof(Type));
        std::FILE *ifp = std::fopen(paths[i].data(), "rb");
        if(!ifp) throw std::runtime_error("Failed to open");
        if(std::fread(s.data(), 1, s.size(), ifp) != s.size()) throw std::runtime_error("Failure in reading");
        std::fclose(ifp);
    }
    std::fprintf(stderr, "loaded sketches\n");
    assert(std::all_of(sketches.begin() + 1, sketches.end(), [&](const auto &x) {
        return x.size() == (&x)[-1].size();
    }));
    blaze::DynamicMatrix<float> distmat(sketches.size(), sketches.size());
    const size_t sketchsz = sketches.front().size();
    std::FILE *rawfp = std::fopen("distmat.raw.float32.np", "wb");
    if(!rawfp) throw 1;
    for(size_t i = 0; i < sketches.size(); ++i) {
        distmat(i, i) = sketches[i].size();
        auto idat = sketches[i].data();
        OMP_PFOR
        for(size_t j = i + 1; j < sketches.size(); ++j) {
            distmat(i, j) = distmat(j, i) = regmatch::count(idat, sketches[j].data(), sketchsz);
        }
        std::fprintf(stderr, "Completed row %zu/%zu\n", i + 1, sketches.size());
    }
    for(size_t i = 0; i < sketches.size(); ++i) {
        auto r(row(distmat, i, blaze::unchecked));
        std::fwrite(r.data(), sizeof(float), r.size(), rawfp);
    }
    std::fclose(rawfp);
}
