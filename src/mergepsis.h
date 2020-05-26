#pragma once
#include "parsepsi.h"
#include <set>
#include "sketch/flat_hash_map/flat_hash_map.hpp"


//#pragma omp declare reduction (merge : ska::flat_hash_map<std::string, blaze::DynamicVector<float>> : update(omp_out, omp_in))
//

void getlabel(std::string &ret, char *s) {
    char *os = s;
    if(!(s = std::strchr(s, ':'))) goto fail;
    if(!(s = std::strchr(s + 1, ':'))) goto fail;
    if(!(s = std::strchr(s + 1, ':'))) goto fail;
    ret = std::string(os, s);
    return;
    fail: {
        std::cerr << "getlabel failure\n";
        throw std::runtime_error("Failed to get label");
    }
}

ska::flat_hash_map<std::string, blaze::DynamicVector<float>> file2lines(std::string path, bool leafcutter, bool normalize=true) {
    if(normalize && !leafcutter) {
        const char *msg = "Can't normalize PSI data.";
        std::cerr << msg << '\n';
        std::exit(1);
        throw std::runtime_error(msg);
    }
    ska::flat_hash_map<std::string, blaze::DynamicVector<float>> ret;
    gzFile fp = gzopen(path.data(), "rb");
    if(!fp) {
        std::cerr << "Failed to open " << path << " for reading\n";
        throw std::runtime_error("Failed to open");
    }
    std::vector<unsigned> offsets;
    std::vector<char> buf(1 << 20);
    if(!gzgets(fp, buf.data(), buf.size())) {
        std::cerr << "Failed to get first line\n";
        throw 2;
    }
    const int n_other_fields = leafcutter ? 1: 12;
    auto dosplit = [&]() {ks::split(buf.data(), leafcutter ? ' ': ',', offsets);};
    dosplit();
    size_t vsize = offsets.size() - n_other_fields;
    std::fprintf(stderr, "[%s] Number of samples: %zu\n", path.data(), vsize);
    std::string label;
    while(gzgets(fp, buf.data(), buf.size())) {
        dosplit();
        assert(offsets.size() == vsize + n_other_fields);
        getlabel(label, buf.data());
        blaze::DynamicVector<float> key(vsize);
        char *s;
        for(size_t i = n_other_fields; i < offsets.size(); ++i) {
            const auto val = std::strtod(buf.data() + offsets[i], &s);
            auto &ref = key[i - n_other_fields];
            if(!normalize || !val) ref = val;
            else {
                assert(*s == '/');
                ref = val / std::atof(++s);
            }
        }
        ret[label] = std::move(key);
    }
    gzclose(fp);
    return ret;
}

blaze::DynamicMatrix<float> files2master(const std::vector<std::string> &paths, bool leafcutter, bool normalize=true) {
    std::vector<ska::flat_hash_map<std::string, blaze::DynamicVector<float>>> collections(paths.size());
    #pragma omp parallel for
    for(size_t i = 0; i < paths.size(); ++i) {
        collections[i] = file2lines(paths[i], leafcutter, normalize);
    }
    ska::flat_hash_map<std::string, uint32_t> labeler;
    {
        std::set<std::string> keys;
        for(const auto &col: collections)
            for(const auto &pair: col)
                keys.insert(pair.first);
        size_t i = 0;
        for(const auto &key: keys)
            labeler[key] = i++;
    }
    size_t total_samples = std::accumulate(collections.begin(), collections.end(), size_t(0), [](auto csum, const auto &x) {return csum + x.begin()->second.size();});
    std::fprintf(stderr, "Number of keys: %zu. samples: %zu\n", labeler.size(), total_samples);
    blaze::DynamicMatrix<float> fulldata(labeler.size(), total_samples);
    std::vector<size_t> sizes;
    fulldata = 0.;
    size_t sample_index = 0;
    OMP_PFOR
    for(size_t i = 0; i < collections.size(); ++i) {
#if 1
        std::fprintf(stderr, "sample index %zu with i = %zu\n", sample_index, i);
        for(const auto &pair: collections[i]) {
            auto id = labeler.at(pair.first);
            subvector(row(fulldata, id), sample_index, pair.second.size()) = trans(pair.second);
        }
        auto csz = collections[i].begin()->second.size();
        sample_index += csz;
        sizes.push_back(csz);
#else
        std::fprintf(stderr, "sample index %zu with i = %zu\n", sizesums[i], i);
        for(const auto &pair: collections[i]) {
            auto ind = labeler.at(pair.first);
            std::fprintf(stderr, "ind: %u\n", ind);
            auto r(row(fulldata, ind));
            std::fprintf(stderr, "second size: %zu\n", pair.second.size());
            subvector(r, sizesums[i], pair.second.size()) = trans(pair.second);
        }
#endif
    }
    for(size_t i = 0; i < sizes.size(); ++i) {
        std::fprintf(stderr, "File %s had %zu entries\n", paths[i].data(), sizes[i]);
    }
    transpose(fulldata);
    return fulldata;
}
