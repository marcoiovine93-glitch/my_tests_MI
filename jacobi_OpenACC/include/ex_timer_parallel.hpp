/**
*  USE THIS AS
*  {CSimple_timer t("WHATEVER YOU ARE TIMING");
*        THE CODE YOU ARE TIMING
*    }
* 
*  THEN AT THE END OF MAIN PUT
*  CSimple_timer::print_timing_results();   
*/

#ifndef TIMER_L_NEW_SPEC
#define TIMER_L_NEW_SPEC

#include <map>
#include <iostream>
#include <chrono>
#include <thread>
#include <string>
#include <cmath>
#include <mpi.h>
#include <fstream>
#include <iomanip>

using time_units=std::chrono::microseconds;

struct TimerData{
    int num_calls = 0;
    double taken_time = 0;
};

std::map<std::string, struct TimerData> table; //We define the table withb the timing data outside the class to make it global and accessible.
//For global variable it is important to avoid having commented initialization of the same variable --> there could be a linking issue!!


class CSimple_timer_parallel{
public:
    std::string timewhat0;
    
    std::chrono::steady_clock::time_point t_start;
    std::chrono::steady_clock::time_point t_end;

    //constructor
    CSimple_timer_parallel(const std::string&);

    //destructor
    ~CSimple_timer_parallel();
    
    //here comes important new info: static class functions - they can be called without the object of the class
    static void print_timing_results(); //It is really important to add the return type!!

    
}; //of class


     //destructor
     CSimple_timer_parallel::~CSimple_timer_parallel(){
         //STOP THE CLOCK
         t_end = std::chrono::steady_clock::now();
 
         //CALCULATE DURATION
         auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();
 
         //INSERT THAT INTO YOUR TABLE
         if(table.find(timewhat0) != table.end()){
              table[timewhat0].taken_time = table[timewhat0].taken_time + duration ; //We increase the num. of calls of the current function called.
         }else{
              table[timewhat0].taken_time = duration ;
         }

         //MPI_Finalize();
     }//destructor


     //here comes important new info: static class functions - they can be called without the object of the class
     void CSimple_timer_parallel::print_timing_results(){
         /*for(const auto& [func, timer] : table) {
             std::cout<<"Function: "<<func<<", Time: "<<timer.taken_time<< " milliseconds, Calls: "<<timer.num_calls<<std::endl;
         }*/
         
         int w_size, rank;
         MPI_Comm_rank(MPI_COMM_WORLD, &rank);
         MPI_Comm_size(MPI_COMM_WORLD, &w_size);

         //We retrieve the size of the map (number of the map pairs):
         size_t size_table = table.size();
         //

         //We instantiate and resize the vectors to retrieve the data contained in the map (the hash map is not a contiguous data structure)
         std::vector<double> vect_time;
         std::vector<std::string> vect_str;

        //This is an error: we don't know the size of the data processed by each process!!
        //We need at first to gather the sizes of the data retrieved by the table for each process, then we do a Gatherv on the root process to
        //get the timing data:
         vect_time.resize(size_table);
         vect_str.resize(size_table);
         //
         
        //We put the map elements (separated between function name and timings) in the 2 vectors:
        int i = 0;
        for(const auto& pair : table) {
           vect_time[i] = pair.second.taken_time;
           vect_str[i] = pair.first;
           i++;
        }
        //

        size_t size_time = vect_time.size(); //It can be different for each process, we need to gather it.
        
        int root = 0;
        std::vector<int> v_siz;
        std::vector<double> outp_time;
        long int reicv_size;
        if(rank == root){
            //v_siz.resize(w_size);
            //reicv_size = (long int)vect_str.size() * w_size; //We assume the vect_str is of the same size for every rank
            outp_time.resize(w_size);
        }
	
	for(long int i = 0; i < vect_time.size(); i++){
        	double vect_one = vect_time[i];
		
		MPI_Gather(&vect_one, 1, MPI_DOUBLE, outp_time.data(), 1, MPI_DOUBLE, root, MPI_COMM_WORLD);
        
        	std::vector<double> compar;
         	
        	std::ostringstream filevar;
        	filevar<<"Func_"<<vect_str[i]<<".txt";
        	std::ofstream file(filevar.str());
        	//We print the comparison between process data in the root process:
        	if(rank == root){
                	compar.resize(w_size);
                	for(int j = 0; j < w_size; j++){
                   	    //file<<"Process: "<<i<<std::endl;
                   	    //file<<"Timing: "<<outp_time[j]<<std::endl;
                       	    compar[j] = outp_time[j];
                	}
                  	file<<std::endl;
            
                	//Let's find the minimum and maximum values:
                	double  minimum = compar[0];
                	int proc = 0;
                	double maximum = compar[0];
                	int proc_m = 0;

                	//Variable to check if all the process showed the same time:
                	int equal = 1;
                	//
                	for(int m = 0; m < w_size-1; m++){
                    	    if(compar[m] != compar[m+1]){
                        	equal = 0;
                        	break;
                    	    }
                	}

                	//if(equal == 0){
              	 	    //Minimum:
                   	    for(int i = 1; i < w_size; i++){
                      	        if(compar[i] <= minimum){
                                    minimum = compar[i];
                          	    proc = i;
                      	        }
                   	    }
               
                   	    //Maximum:
                   	    for(int i = 1; i < w_size; i++){
                                if(compar[i] >= maximum){
                                    maximum = compar[i];
                                    proc_m = i;
                      	        }
                   	    }//
 
                   	    //Average:
                   	    double sum = 0;
                   	    for(int k=0; k < w_size; k++){
                                sum += compar[k];
                   	    }
                   	    //
			    
                   	    double average = sum/w_size;
                                 
                   	    file<<"Comparison between processes: "<<std::endl;
                   	    file<<"Minimum: \n"<<" Process: "<<proc<<" , Time: "<<minimum<<std::endl;
                   	    file<<"Maximum: \n"<<" Process: "<<proc_m<<" , Time: "<<maximum<<std::endl;
                   	    file<<"Average: \n"<<" Time: "<<average<<std::endl;
			    file<<"Sum: \n"<<sum<<std::endl;
                        /*}else{
                            file<<"The processes had the same exec. time."<<std::endl;
                        }//end if-else variable equal
			*/
                } //end if for printing in the root. 
            
		file.close();
	
	} //End of for loop to generate a file for each Function called.


      } //static print function : we can call it without instantiating a class object!
     

      //constructor
      CSimple_timer_parallel::CSimple_timer_parallel( const std::string & curr_funct){
          //control in the time when the constructor and destructor are called, it could happen slightly before the main execution and cause issues.
          
          //SET WHAT WE ARE TIMING FROM THE PASSED PARAMETER
          timewhat0 = curr_funct;
 
          //START THE CLOCKS
          t_start = std::chrono::steady_clock::now();
 
          if(table.find(curr_funct) != table.end()){
              table[curr_funct].num_calls = table[curr_funct].num_calls + 1 ; //We increase the number of calls of the current function called.
          }else{
              TimerData newData;
              newData.num_calls = 1;
              table.insert(std::make_pair(curr_funct, newData));
          }
          
          /*
          //We retrieve the size of the map (number of the map pairs):
          size_t size_table = table.size();
          //

          //We instantiate and resize the vectors to retrieve the data contained in the map (the hash map is not a contiguous data structure)
          std::vector<int> vect_str;
          std::vector<double> vect_time;
          vect_str.resize(size_table);
          vect_time.resize(size_table);
          //
          
          //We put the map elements (separated between function name and timings) in the 2 vectors:
          int i = 0;
          for(const auto& pair : table) {
              vect_str[i] = pair.second.num_calls;
              vect_time[i] = pair.second.taken_time;
              i++;
          }
          //
          
          //We do the gathering for the vector processed by each processor in order to print all the results from the root process:
          //Of course we will do the gathering for both the vectors containing the informations needed.
          int root = 0;
          //We instantiate and resize 2 rceiving vector buffers: one for thr number of calls, and one for the timing:
          std::vector<int> reicv_calls;
          std::vector<double> reicv_time;

          if(rank == root){
              reicv_calls.resize(size_table);
              reicv_time.resize(size_table);
          }

          MPI_Gather(vect_str.data(),(int)vect_str.size(), MPI_INT, reicv_calls.data(),(int)reicv_calls.size(), MPI_INT, root, MPI_COMM_WORLD);
          MPI_Gather(vect_time.data(),(int)vect_time.size(), MPI_DOUBLE, reicv_time.data(),(int)reicv_time.size(), MPI_DOUBLE, root, MPI_COMM_WORLD);
          //


          //The size of the num calls and timing vectors is the same, so we can take refer to one of them
          if(rank == root){
            for (size_t j = 0; j < reicv_time.size(); j++) {
                std::cout <<"Process : "<<rank<<", num. calls: "<<reicv_time[j]<<" , "<<reicv_calls[j]<<std::endl;
            }
            std::cout<<std::endl;
          }
          // */


          //std::cout<<"Constructor called"<<std::endl;
 
      } //constructor

#endif
