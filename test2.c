#include "llmc/pfloat.h"

std::vector<pfloat>* get_pfloat_vec(size_t n) {
    std::vector<pfloat>* vec = new std::vector<pfloat>(n);
    return vec;
}

int main (int argc, char** argv) {

    int n = atoi(argv[1]);
    
    std::vector<pfloat>* v = get_pfloat_vec(n);
    v->at(1) = 0.5;
    v->at(3) = 0.25;
    v->at(2) = v->at(1)+v->at(3);
    printf("v elements:\n");
    for (int i=0; i<n; i++) {
	printf("v->at(%d) = %.8e\n",i,v->at(i).convert_to<float>());
    }
    printf("p it:\n");
    pfloat* p = v->data();
    p[3] = p[2] + p[1];
    for (int i=0; i<n; i++) {
	printf("p (%d) = %.8e\n",i,p[i].convert_to<float>());
    }
    return 0;
}
