#include "blaze/Math.h"
#include "kspp/ks.h"

size_t countlines(std::string path) {
    gzFile fp = gzopen(path.data(), "rb");
    if(!fp) throw 2;
    size_t ret = 0;
    for(int c;(c = gzgetc(fp)) != EOF; ret += c == '\n');
    gzclose(fp);
    return ret;
}

template<typename FT=float>
blaze::DynamicMatrix<FT> leafcutter_parsepsi(gzFile fp, unsigned numlines, int sep=' ') {
    if(!fp) throw 3;
    std::vector<unsigned> ids;
    char buf[1 << 16];
    gzgets(fp, buf, sizeof(buf));
    std::vector<int> offsets;
    ks::split(buf, sep, offsets);
    unsigned nsamples = offsets.size() - 1;
    std::fprintf(stderr, "n samples: %u\n", nsamples);
    blaze::DynamicMatrix<FT> ret(numlines, nsamples);

    size_t rownum = 0;
    while(gzgets(fp, buf, sizeof(buf))) {
        ids.push_back(std::atoi(buf));
        ks::split(buf, sep, offsets);
        for(size_t i = 1; i < offsets.size(); ++i) {
            char *s;
            double num = std::strtod(buf + offsets[i], &s);
            ret(rownum, i - 1) = num / std::atof(s + 1);
        }
        rownum++;
    }
    std::fprintf(stderr, "row num: %zu. nulines: %u\n", rownum, numlines);
    assert(rownum == numlines);
    transpose(ret);
    return ret;
}

template<typename FT=float>
blaze::DynamicMatrix<FT> parsepsi(gzFile fp, unsigned numlines, int sep=',') {
    if(!fp) throw 3;
    std::vector<unsigned> ids;
    char buf[1 << 16];
    gzgets(fp, buf, sizeof(buf));
    std::vector<int> offsets;
    ks::split(buf, sep, offsets);
    unsigned nsamples = offsets.size() - 12;
    std::fprintf(stderr, "n samples: %u\n", nsamples);
    blaze::DynamicMatrix<FT> ret(numlines, nsamples);

    size_t rownum = 0;
    while(gzgets(fp, buf, sizeof(buf))) {
        ids.push_back(std::atoi(buf));
        ks::split(buf, sep, offsets);
        for(size_t i = 12; i < offsets.size(); ++i) {
            ret(rownum, i - 12) = std::atof(buf + offsets[i]);
            assert(std::strlen(buf + offsets[i]));
        }
        rownum++;
    }
    std::fprintf(stderr, "row num: %zu. nulines: %u\n", rownum, numlines);
    assert(rownum == numlines);
    transpose(ret);
    return ret;
}

template<typename FT=float>
auto parsepsi(std::string path) {
    gzFile fp = gzopen(path.data(), "rb");
    if(!fp) throw 4;
    auto ret = parsepsi(fp, countlines(path) - 1); // Subtract 1 for header
    gzclose(fp);
    return ret;
}

template<typename FT=float>
auto leafcutter_parsepsi(std::string path) {
    gzFile fp = gzopen(path.data(), "rb");
    if(!fp) throw 4;
    auto ret = leafcutter_parsepsi(fp, countlines(path) - 1); // Subtract 1 for header
    gzclose(fp);
    return ret;
}
