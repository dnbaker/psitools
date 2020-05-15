#pragma once
#include "parsepsi.h"
#include <set>
#include "sketch/flat_hash_map/flat_hash_map.hpp"


//#pragma omp declare reduction (merge : ska::flat_hash_map<std::string, blaze::DynamicVector<float>> : update(omp_out, omp_in))

ska::flat_hash_map<std::string, blaze::DynamicVector<float>> file2lines(std::string path, bool leafcutter) {
    ska::flat_hash_map<std::string, blaze::DynamicVector<float>> ret;
    gzFile fp = gzopen(path.data(), "rb");
    if(!fp) throw std::runtime_error("Failed to open");
    gzclose(fp);
    std::vector<unsigned> offsets;
    char buf[1 << 16];
    if(!gzgets(fp, buf, sizeof(buf))) throw 2;
    const int n_other_fields = leafcutter ? 1: 12;
    auto dosplit = [&]() {
        if(leafcutter) {
            ks::split(buf, ' ', offsets);
        } else {
            ks::split(buf, ',', offsets);
        }
    };
    dosplit();
    size_t vsize = offsets.size() - n_other_fields;
    std::string label;
    while(gzgets(fp, buf, sizeof(buf))) {
        dosplit();
        label = std::string(buf);
        blaze::DynamicVector<float> key(vsize);
        for(size_t i = n_other_fields; i < offsets.size(); ++i) {
            key[i - n_other_fields] = std::atof(buf + offsets[i]);
        }
        ret[label] = std::move(key);
    }
    return ret;
}

blaze::DynamicMatrix<float> files2master(const std::vector<std::string> &paths, bool leafcutter) {
    std::vector<ska::flat_hash_map<std::string, blaze::DynamicVector<float>>> collections(paths.size());
    #pragma omp parallel for
    for(size_t i = 0; i < paths.size(); ++i) {
        collections[i] = file2lines(paths[i], leafcutter);
    }
    ska::flat_hash_map<std::string, uint32_t> labeler;
    std::vector<std::string> keyv;
    {
        std::set<std::string> keys;
        for(const auto &col: collections)
            for(const auto &pair: col)
                keys.insert(pair.first);
        keyv.assign(keys.begin(), keys.end());
        for(size_t i = 0; i < keyv.size(); ++i)
            labeler[keyv[i]] = i;
    }
    size_t total_samples = std::accumulate(collections.begin(), collections.end(), size_t(0), [](auto csum, const auto &x) {return csum + x.size();});
    blaze::DynamicMatrix<float> fulldata(keyv.size(), total_samples);
    std::vector<size_t> sizes;
    fulldata = 0.;
    size_t sample_index = 0;
    for(size_t i = 0; i < collections.size(); ++i) {
        const auto &col = collections[i];
        for(const auto &pair: col) {
            auto id = labeler.at(pair.first);
            subvector(row(fulldata, id), sample_index, pair.second.size()) = trans(pair.second);
        }
        auto csz = col.begin()->second.size();
        sample_index += csz;
        sizes.push_back(csz);
    }
    for(size_t i = 0; i < sizes.size(); ++i) {
        std::fprintf(stderr, "File %s had %zu entries\n", paths[i].data(), sizes[i]);
    }
    return fulldata;
}
