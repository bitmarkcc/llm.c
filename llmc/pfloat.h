#ifndef PFLOAT_H
#define PFLOAT_H

#include <boost/multiprecision/mpfr.hpp>

typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<8> > pfloat;
typedef boost::multiprecision::number<boost::multiprecision::mpfr_float_backend<17> > pdouble;

#endif
