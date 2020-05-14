#include "parsepsi.h"
#include "blaze/Util.h"
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <getopt.h>


int main(int argc, char **argv) {
    blaze::DynamicMatrix<float> mat;
    int c;
    bool leafcutter = false;
    while((c = getopt(argc, argv, "l:h?")) >= 0) {
        switch(c) {
            case 'l': leafcutter = true; break;
            case 'h': std::fprintf(stderr, "usage: %s <opts> inputfile blazeout\n-l: Leafcutter mode\n-h: Usage.\n", argv[0]);
                      std::exit(1);
        }
    }
    if(optind + 2 >= argc) {
        std::fprintf(stderr, "usage: %s <opts> inputfile blazeout\n-l: Leafcutter mode\n-h: Usage.\n", argv[0]);
        std::exit(1);
    }
    mat = parsepsi<>(argv[optind + 1]);
    blaze::Archive<std::ofstream> outarch(argv[optind + 2]);
    outarch << mat;
    std::fprintf(stderr, "Wrote matrix to blaze dump. %zu rows, %zu cols\n", mat.rows(), mat.columns());
}
