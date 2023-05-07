
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include "nanoflann.hpp"
#include <omp.h>

using namespace std;
constexpr int SAMPLES_DIM = 3;
//typedef double ScalarType;
//typedef Eigen::Matrix<ScalarType, 3, 1, 0, 3, 1> EigenVector3;
//typedef Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> VectorX;
//#define block_vector(a) block<3,1>(3*(a), 0)


template <typename Der>
void generateRandomPointCloud(
    Eigen::MatrixBase<Der>& mat, const size_t N, const size_t dim,
    const typename Der::Scalar max_range = 10)
{
    std::cout << "Generating " << N << " random points...";
    mat.resize(N, dim);
    for (size_t i = 0; i < N; i++)
        for (size_t d = 0; d < dim; d++)
            mat(i, d) =
            max_range * (rand() % 1000) / typename Der::Scalar(1000);
    std::cout << "done\n";
}



int main(int argc, char** argv)
{
    //Eigen::initParallel();
    // setup
    int nSamples = 87;
    int dim = SAMPLES_DIM;

    using matrix3t = Eigen::Matrix<double, Eigen::Dynamic, 3>;

    matrix3t mat(nSamples, dim);

    const double max_range = 20;

    // Generate points:
    generateRandomPointCloud(mat, nSamples, dim, max_range);
    //std::cout << mat << std::endl;
    // Query point:

    //using my_kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<matrix3t>;
    using my_kd_tree_t = nanoflann::KDTreeEigenMatrixAdaptor<matrix3t>;
    my_kd_tree_t mat_index(dim, std::cref(mat), 10 /* max leaf */);

    // do a knn search
    const size_t        num_results = nSamples;
    std::vector<size_t> ret_indexes(num_results);
    std::vector<double>  out_dists_sqr(num_results);
    nanoflann::KNNResultSet<double> resultSet(num_results);


    Eigen::MatrixXi checkmat;
    checkmat.resize(mat.rows(), mat.rows());
    checkmat.setIdentity();
    // print output
    //VectorX ESforce;
    //ESforce.resize(mat.rows()*3,1);
    //ESforce.setZero();
    int count = 0;




#pragma omp parallel for 
    for (auto j = 0; j < mat.rows(); j++) {
        std::vector<double> query_pt = { mat(j,0), mat(j,1), mat(j,2) };
        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        mat_index.index_->findNeighbors(resultSet, &query_pt[0]);

        for (int i = 0; i < resultSet.size(); i++)
        {
            if (checkmat(j, ret_indexes[i]) == 0) {

                checkmat(ret_indexes[i], j) += 1;
                checkmat(j, ret_indexes[i]) += 1;
                count += 1;
            }
                
        }
    }
    

    std::cout << "expected computation = 3741, actual computation = " << count << std::endl;




    return 0;

}
