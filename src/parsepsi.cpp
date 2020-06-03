#include "parsepsi.h"
#include "blaze/Util.h"
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <getopt.h>


int main(int argc, char **argv) {
    blaze::DynamicMatrix<float> mat;
    blaze::DynamicMatrix<double> dmat;
    int c;
    bool use_double = false;
    bool leafcutter = false;
    while((c = getopt(argc, argv, "dl:h?")) >= 0) {
        switch(c) {
            case 'l': leafcutter = true; break;
            case 'd': use_double = true; break;
            case 'h': std::fprintf(stderr, "usage: %s <opts> inputfile blazeout\n-l: Leafcutter mode\n-h: Usage.\n", argv[0]);
                      std::exit(1);
        }
    }
    if(optind + 2 >= argc) {
        std::fprintf(stderr, "usage: %s <opts> inputfile blazeout\n-l: Leafcutter mode\n-h: Usage.\n-d: use doubles (vs float)\n", argv[0]);
        std::exit(1);
    }
    if(use_double) {
        if(leafcutter) {
            mat = leafcutter_parsepsi<>(argv[optind + 1]);
        } else {
            mat = parsepsi<>(argv[optind + 1]);
        }
        blaze::Archive<std::ofstream> outarch(argv[optind + 2]);
        outarch << mat;
        std::fprintf(stderr, "Wrote matrix to blaze dump. %zu rows, %zu cols\n", mat.rows(), mat.columns());
    } else {
        if(leafcutter) {
            dmat = leafcutter_parsepsi<double>(argv[optind + 1]);
        } else {
            dmat = parsepsi<double>(argv[optind + 1]);
        }
        blaze::Archive<std::ofstream> outarch(argv[optind + 2]);
        outarch << dmat;
        std::fprintf(stderr, "Wrote matrix to blaze dump. %zu rows, %zu cols\n", dmat.rows(), dmat.columns());
    }
}
