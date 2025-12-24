/**
The current Class has the aim of initializing and creating the Grid. All the member variables and functions have accessibility set as public.
Two vectors (dynamically allocated memory corresponding to pointer to nodes provided by the previous iteration and pointer to nodes of the current
iteration. The class presents a template for its objects type. The constructor has 2 parameters: the grid size (it is indeed the square root of the 
real size of the grid) and the boundary conditions, that are passed as a Template function. It is important to observe that the size parameter
is passed by value, not by reference. In the constructor, both vectors are resized and for both we implement the boundary conditions.
Eventually, we have the print function. The indices for the for loops are managed so that the rows with the lower value of the i index (vertical
index) are at the bottom in order to have a y axis orientation from down to up.
**/

#ifndef CMESH_LOL
#define CMESH_LOL

#include <vector>
#include <iostream>
#include <cmath>
#include <fstream>
#include <mpi.h>
#include <cmath>
#include <iomanip>
#include "ex_timer_parallel.hpp"
//#include <omp.h>
#include <openacc.h> //We include the header corresponding to openacc.
#include <cuda_runtime.h>


template <typename T> class CMesh{
    public:
    int size;
    int size_new;

    //MPI params:
    int rank, w_size;
    //
    
    //We define the needed parameters to treat both the case with a n. of processes which gives no rest to respect to the matrix size and the 
    //case with no rest:
    int nloc, rest, offset, index_offset, prev, next; 
    //

    //We consider a start and end index because for the computation we need to change the first and last computed row depending on the rank
    //(the first and last rank need less rows due to the boundaries)
    //These indexes are needed because we consider the chosen value N as the total size of the matrix (we don't consider it as the matrix of
    //the inernal points without boundaries:
    int start_ind, end_ind;
    //

    //We introduce ghost lines and the send and receive buffers to send and receive the local boundary rows betw. ranks:
    std::vector<T> ghost_up;
    std::vector<T> ghost_down;
    std::vector<T> send_buff_loc_prev;
    std::vector<T> send_buff_loc_next;
    //

    std::vector<T> buff_print;
    std::vector<T> displs;
    std::vector<T> recv_print;
    
    //We define the old and new field for jacobi iterative method:
    std::vector<T> field;
    std::vector<T> new_field;
    //
    
    //We instantiate pointers to communicate between Host and Device and they need to be equivalent to field and new field:
    T* field1;
    T* new_field1;
    T* ghost_up1;
    T* ghost_down1;
    T* send_buff_loc_prev1;
    T* send_buff_loc_next1;

    void boundary_conditions(std::vector<T>& field);

    // We introduce 2 templates: one is for the variable type, the other (F) is for the function --> this notation let us 
    // call the constructor making the compiler to understand that the second argument is related to the function introduced
    // for the boudary conditions.
    //template <typename F>
    CMesh(const int& N/*, F& boundary_conditions*/); //Constructor: WARNING!: we pass the size by value, not by reference!!! Passing by reference causes                                                  //a lot of issues!!
    void print_funct(std::ofstream& os); //Print function
    
    void jacobi(const size_t& max_steps, const size_t& PrintInterval);
    
    void local_boundary(std::vector<T>& fiel);

};



//Constructor:
template <typename T>
//template <typename F> //In this case we introduce a Template Function.
CMesh<T>::CMesh(const int& N)
{
    size = N ; 
    size_new = N ;

    //We set the MPI environment parameters:
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &w_size);
    //
    //size = N ;
    //size_new = N ;

    nloc = size/w_size;
    rest = size%w_size;
    
    //We set an index offset to have the conversion between local i index and global i index in order to have the correct change
    //of the boundary condition on the left side of the domain (which will depend on the process rank):
    index_offset = nloc*rank;
    //

    if(rank < rest){
        nloc += 1;
        offset = rank;
    }else{
        offset = rest;
    }
    
    //std::cout<<"Constructor first debug"<<std::endl;
    //std::cout<<"Nloc: "<<nloc<<" Rank: "<<rank<<std::endl;
    // RESIZE FIELDS HERE AND SET THE SIZE
    field.resize(size*nloc); // If we don't do the resize, what happens?--> it is like not initialized!   
    new_field.resize(size*nloc); 
    
    //Pointers initialization for GPU:
    field1 = field.data();
    new_field1 = new_field.data();
    ///
    
    ghost_up.resize(size,0);
    ghost_down.resize(size,0);
    
    //Pointers initialization for GPU:
    ghost_up1 = ghost_up.data();
    ghost_down1 = ghost_down.data();
    //

    send_buff_loc_prev.resize(size,0);
    send_buff_loc_next.resize(size,0);
    
    //Pointers initialization for GPU:
    send_buff_loc_prev1 = send_buff_loc_prev.data();
    send_buff_loc_next1 = send_buff_loc_next.data();
    ///
    
    //WARNING!!!!!!!!!!!DO NOT ADD THIS LINE WHEN CHECKING THE CODE PERFORMANCE! IT ALLOCATES A VECTOR OF SEVERAL ELEMENTS WHICH BREAKS THE MEMORY. 
    //buff_print.resize(size*size,0);
    
    start_ind = (rank == 0) ? 1 : 0 ;
    end_ind = (rank == (w_size - 1))? (nloc - 1) : nloc;
    ///std::cout<<"Constructor sec debug"<<std::endl;
    
    prev = (rank == 0) ? MPI_PROC_NULL : rank-1 ;
    next = (rank == (w_size - 1)) ? MPI_PROC_NULL : rank+1;

    //set up the boundary conditions
    //even though in this toy we could just call a lambda inside the class
    // we want a more general approach in case we want to change the conditions    
    
    {CSimple_timer_parallel t{"Initialization16:"};
    	boundary_conditions(field); //It is so important to initialize both the fields, otherwise the resizing will give a null
    	boundary_conditions(new_field); //elements new_field
    }

    //std::cout<<"Constructor 3rd  debug"<<std::endl;

} // constuctor


template <typename T> //Print function
void CMesh<T>::print_funct(std::ofstream& os){
    std::vector<int> all_nloc(w_size);
    MPI_Gather(&nloc, 1, MPI_INT, all_nloc.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank == 0){
        for(int i = 0; i < (int)nloc; i++){ //We print with the rows direction from bottom to top.
            for(int j = 0; j < (int)size; j++){
               os<<field[j + size*i]<<" ";
            }
            os<<std::endl;
        }
        for(int s = 1; s < w_size; s++){
            int rows = all_nloc[s];
            std::vector<T> buff_print(rows * size);
            MPI_Recv(buff_print.data(), rows * size, MPI_DOUBLE, s, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            for(int i = 0; i < rows; i++){ //We print with the rows direction from bottom to top.
                for(int j = 0; j < (int)size; j++){
                    os<<buff_print[j + size*i]<<" ";
                }
                os<<std::endl;
            }
        }
    }else{
        MPI_Send(field.data(), nloc*size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

} //print function


template <typename T>
void CMesh<T>::jacobi(const size_t& max_steps, const size_t& PrintInterval){
    //size_t sizen = m.size;
    
    //We introduce three loops: one for the number of iterations, and 2 loops for the calculation of the solution. The computation is done only
    //in the internal points of the dominium, to guarantee a constant value of the boundary values within the iteration process:
    
    //We should copy data from host to device:
    
    //We should define local variables, otherwise the whole CMesh object is passed to the device and this cannot happen:
    int size_d = size;
    int nloc_d = nloc;
    int rank_d = rank;
    int w_size_d = w_size;
    int start_ind_d = start_ind;
    int end_ind_d = end_ind;
    
    double* field_d = field1;
    double* new_field_d = new_field1;
    double* ghost_up_d = ghost_up1;
    double* ghost_dn_d = ghost_down1;

    #pragma acc data copy(field1[0:size_d*nloc_d], new_field1[0:size_d*nloc_d], send_buff_loc_prev1[0:size_d], send_buff_loc_next1[0:size_d], ghost_down1[0:size_d], ghost_up1[0:size_d])
    {
	for(int k = 0; k < (int)max_steps; k++){
        //We introduce an ofstream to possibly print the field values in the nodes.
                /*
	    	{CSimple_timer_parallel t{"Communication16:"};
	    		local_boundary(field);
	    	}*/
            //We do send receive among submatrices from GPU:
            #pragma acc host_data use_device(field1, new_field1, send_buff_loc_prev1, send_buff_loc_next1, ghost_down1, ghost_up1)
	    {
		///////////
		for(int i=0; i < size; i++){
          		send_buff_loc_prev[i] = field[i];
      		}

      		for(int i=0; i < size; i++){
          		send_buff_loc_next[i] = field[size*(nloc-1) + i];;
      		}

      		//We introduce 2 Sendrecv: one to send up and one to send down
      		//Send up:
      		MPI_Sendrecv(send_buff_loc_prev.data(),size,MPI_DOUBLE,prev,0,ghost_down.data(),size,MPI_DOUBLE,next,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

      		//Send down:
      		MPI_Sendrecv(send_buff_loc_next.data(),size,MPI_DOUBLE,next,1,ghost_up.data(),size,MPI_DOUBLE,prev,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);

	    }/////

	    	//{CSimple_timer_parallel t{"Computation16:"};
                        #pragma acc parallel loop collapse(2) present(field_d, new_field_d, ghost_up_d, ghost_down_d)
			for(int i = start_ind_d; i < end_ind_d; i++){//When I amon the internal ranks,I should take also the first row of the sub-matrix!
                		for(int j = 1; j < (size_d-1); j++){
                        		//We define an up and down value because they will be dependant on the process rank (the 0 and w_size-1 processes                                          don't //have one of the ghost lines and need to be treated separately)
					double field_up1 = (i == 0 && rank_d != 0) ? ghost_up_d[j] : field_d[(i-1)*size_d + j] ;
                        		double field_down1 = (i == (nloc_d-1) && rank_d != (w_size_d-1)) ? ghost_dn_d[j] : field_d[(i+1)*size_d + j] ;
                        		new_field_d[i*size + j] = 0.25*(field_up1
                                                    + field_down1
                                                    + field_d[i*size + j+1]
                                                    + field_d[i*size + j-1]);
                		}
            		}
	    	//}
                
       		//std::cout<<"We are before the swap"<<std::endl;   
		//#pragma acc host_data use_device(field, new_field)
		//{
		    //We swap the pointers on the Device:
		  //  field.swap(new_field); //The two vectors, one for the current iteration and one for the previous one, are swapped in order to make the 
                                 //pointers of the new one be associated to the old field and viceversa.
	        //}
		
		//We swap pointers on the Host:
		//field.swap(new_field);
		
		T* temp = field1;
		field1 = new_field1;
		new_field1 = temp;

		//std::cout<<"We are after the swap"<<std::endl;
        	if(k%PrintInterval == 0){ //we print if the print interval is a number for which thr current n. of iteration is dividable
            	//We print the 
            	std::ostringstream filevar;
            	filevar << "iterPar_" << std::setw(2) << std::setfill('0') << k+10 << ".dat";
            	std::ofstream file(filevar.str());
            	print_funct(file);
        	} //if statement*/
        //std::cout<<"We are after after the swap for iteration: "<<k<<std::endl;
    	} //for loop to print in file
   
    }
}

template <typename T>
void CMesh<T>::boundary_conditions(std::vector<T>& field){
    T init_value = 100;
    T intern_init_value = 0.5;

    /*std::cin>>init_value;
    std::cin.ignore();
    std::cin>>intern_init_value;*/

    //long int newsize = sqrt((int)field.size());
    for(long int i = 0; i < size*nloc; i++){ // We set the value in the internal points of the domain and also on the boundary points
        field[i] = intern_init_value; //In the next for cycle the boundary values are updated with the correct required values.
    }
     
    if(rank == 0){
        for(long int j = 0; j < size; j++){
            field[j] = -(init_value/(size - 1))*j + init_value; //Lower boundary points values
        }
    }
    
    if(rank == (w_size - 1)){
	for(long int j = 0; j < size; j++){
           // std::cout<<"Inside the rank 1 previous"<<std::endl;
            field[size*(nloc-1) + j] = 0; //Upper boundary
           // std::cout<<"Inside the rank 1"<<std::endl;
        }
    }
    
    for(int g = 0; g < nloc; g++){
        field[g*size + (size-1)] = 0; //Rigth boundary
        //It is important to observe the global index for the left boundary:
        field[g*size] = -(init_value/(size - 1))*(g + index_offset + offset)  + init_value; //Left boundary
        //
    }
    
    //Print of the matrix for debugging:
        /*if(rank == 0){
        std::cout<<"Rank: "<<rank<<std::endl;
        for(int r = 0; r < nloc; r++){
            for(long int s = 0; s < size; s++){
                std::cout<<field[r*size + s]<<" ";
            }
            std::cout<<std::endl;
        }
        }*/

}

//In the following function we define the communication between consecutive ranks in order toexchange local boundary rows on ghost lines:
template <typename T>
void CMesh<T>::local_boundary(std::vector<T>& field){
      //int sizeloc = (int)size;
      
      for(int i=0; i < size; i++){
          send_buff_loc_prev[i] = field[i];
      }
      
      for(int i=0; i < size; i++){
          send_buff_loc_next[i] = field[size*(nloc-1) + i];;
      }

      //We introduce 2 Sendrecv: one to send up and one to send down
      //Send up:
      MPI_Sendrecv(send_buff_loc_prev.data(),
                   size,
                   MPI_DOUBLE,
                   prev,
                   0,
                   ghost_down.data(),
                   size,
                   MPI_DOUBLE,
                   next,
                   0,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
      
      //Send down:
      MPI_Sendrecv(send_buff_loc_next.data(),
                   size,
                   MPI_DOUBLE,
                   next,
                   1,
                   ghost_up.data(),
                   size,
                   MPI_DOUBLE,
                   prev,
                   1,
                   MPI_COMM_WORLD,
                   MPI_STATUS_IGNORE);
      //
}

#endif





