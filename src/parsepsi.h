#pragma once
#include "blaze/Math.h"
#include "kspp/ks.h"

size_t countlines(std::string path) {
    gzFile fp = gzopen(path.data(), "rb");
    if(!fp) throw std::runtime_error("Couldn't count lines");
    size_t ret = 0;
    for(int c;(c = gzgetc(fp)) != EOF; ret += c == '\n');
    gzclose(fp);
    return ret;
}

template<typename FT=float>
blaze::DynamicMatrix<FT> leafcutter_parsepsi(gzFile fp, unsigned numlines, int sep=' ') {
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
            double denom = std::atof(s + 1);
            ret(rownum, i - 1) = denom ? num / denom: -1.;
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
    if(!fp) throw std::runtime_error("Couldn't open file to parsepsi");
    std::vector<unsigned> ids;
    char buf[1 << 16];
    if(!gzgets(fp, buf, sizeof(buf))) throw std::runtime_error("Failed to read header");
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
            auto v = std::atof(buf + offsets[i]);
            if(v > 0) v *= .01;
            ret(rownum, i - 12) = v;
            assert(buf && std::strlen(buf + offsets[i]));
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
    if(path.empty()) path = "";
    std::fprintf(stderr, "path: %s\n", path.data());
    gzFile fp = gzopen(path.data(), "rb");
    if(!fp) throw std::runtime_error("Couldn't open file to parsepsi");
    auto ret = parsepsi(fp, countlines(path) - 1); // Subtract 1 for header
    gzclose(fp);
    return ret;
}

template<typename FT=float>
auto leafcutter_parsepsi(std::string path) {
    gzFile fp = gzopen(path.data(), "rb");
    if(!fp) throw std::runtime_error("Couldn't open file to lcparsepsi");
    auto ret = leafcutter_parsepsi(fp, countlines(path) - 1); // Subtract 1 for header
    gzclose(fp);
    return ret;
}
