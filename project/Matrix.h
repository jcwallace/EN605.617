#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <iostream>

class Matrix {
        int num_rows;
        int num_columns;
        int **m;
    public:
        Matrix(int,int);
        Matrix();
        ~Matrix();
        Matrix( const Matrix& );

        inline double& operator()(int x, int y) { return m[x][y]; }
        friend std::ostream& operator<<(std::ostream&, const Matrix&);
};

Matrix operator*(const Matrix&, const Matrix&);

#endif