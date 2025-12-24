/**
The aim of the class is to implement the iterative procedure for solving the equation.
The class has a constructor that has the size of the grid (the square of the real size of the grid) as parameter and that initializes 
in its body (the format is compressed on a single line) of course also the member variable representing the Grid (named mesh in this case).
The member function Jacobi, that will be used of course only after having created an object (instance) of the class in the main,
implements the iterative procedure through 3 for loops:
- 2 for loops are introduced to consider each node of the grid, with indices i and j;
- One loop is introduced for the number of iteration to be executed.
At the line 61, an if statement checks if, at the current iterative step executed, will be necessary to print the grid values in a file.
The printing is done with the use of the corresponding member function acting on mesh passed as argument (object of the class CMesh<T>).
At the end of each iterative step, is performed a swap between the old vector field and the new one, in order to guarantee that the solution
is updated at each step by swapping the pointers.
**/

#include <cmath>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "CMesh.hpp"


template <typename T>
class CSolver{
public:
    size_t size_solv;
    size_t print_solv;
    CMesh<T> mesh;
    
    //Constructor:
    CSolver(const size_t& size_s /*, const size_t& print_s*/);

    void jacobi(CMesh<T>& m, const size_t& max_steps, const size_t& PrintInterval); //Is it necessary to have class member parameters for this 
                                                                                    //function?

};



//Constructor:
template <typename T>
CSolver<T>::CSolver(const size_t& size_s) : size_solv(size_s), mesh(size_s, boundary_conditions<T>){}


template <typename T>
void CSolver<T>::jacobi(CMesh<T>& m, const size_t& max_steps, const size_t& PrintInterval){
    size_t sizen = m.size;
    
    //We introduce three loops: one for the number of iterations, and 2 loops for the calculation of the solution. The computation is done only
    //in the internal points of the dominium, to guarantee a constant value of the boundary values within the iteration process:
    for(int k = 0; k < (int)max_steps; k++){
        //We introduce an ofstream to possibly print the field values in the nodes.
        for(int i = 1; i < (int)sizen-1; i++){
            for(int j = 1; j < (int)sizen-1; j++){
                m.new_field[i*(int)sizen + j] = 0.25*(m.field[(i-1)*(int)sizen + j] + m.field[(i+1)*(int)sizen + j] + m.field[i*(int)sizen + j+1] + m.field[i*(int)sizen + j-1]);
            }
        }

        m.field.swap(m.new_field); //The two vectors, one for the current iteration and one for the previous one, are swapped in order to make the 
                                 //pointers of the new one be associated to the old field and viceversa.
        if(k%PrintInterval == 0){ //we print if the print interval is a number for which thr current n. of iteration is dividable
            //We print the 
            std::ostringstream filevar;
            filevar << "iter_" << std::setw(2) << std::setfill('0') << k+10 << ".dat";
            std::ofstream file(filevar.str());
            m.print_funct(file);
        } //if statement

    } //for loop to print in file
}
