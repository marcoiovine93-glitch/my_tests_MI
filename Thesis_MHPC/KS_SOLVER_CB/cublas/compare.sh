#!/bin/bash
#Script for comparing lines of files .dat

# We set 2 arguments for the script:
ARGS=2
E_BADARGS=85
E_UNREADABLE=86

#If statement to check if the arguments have been passed correctly:
# The line with "eco.." prints in the terminal 
# $0 : contains the script that launchs the current program with its path
# basename : removes the path directories from the $0 Output
# file1 file2 : tells that the script expects 2 arguments 
if [ $# -ne "$ARGS" ]
then
	echo "Usage: 'basename $0' file1 file2"
	exit $E_BADARGS
fi


# If statement to check readability of the files:
# [[]] : bash stronger conditional syntax
if [[ ! -r "$1" || ! -r "$2" ]]
then
	echo "The 2 files must be readable"
	exit $E_UNREADABLE
fi

# We introduce the 2 arguments:
#$1 = matrix_no_batched.dat
#$2 = matrix_batched.dat


# We create the bash variable tol:
tol=1e-14

# THE FOLLOWING BLOCK WORKS ONLY WITH INTEGERS, NOT WITH DECIMALS:
# We compare each line value and check if the difference is under the tol:
# paste: it introduces a mapping between the lines and the variables a and b
# 	assigning to them the values in the files
# $((a - b)) : the double brakets are used to take into account the numerical 
# 	values
# The [ command indicates the test command in bash, so it needs a space before
# The quotes "" are introduced to take into account variable names with spaces
# ${diff#-} takes the absolute value of diff
#paste file1 file2  | while read  -r a b
#do 
#	diff=$((a - b))
#	if [ "${diff#-}" -gt "$tol" ]
#	then
#	       echo "The difference is greater than the fixed tolerance and it is equal to: " $diff
#               exit 1
#	fi
#done 


# CODE FOR DECIMAL VALUES:
# awk -v tol= : it means to start AWK and create an AWK variable called tol
paste "$1" "$2"  | awk -v tol="$tol" '
{
	diff = $1 - $2
	absd = (diff<0) ? -diff : diff

	if (absd > tol) {
		print "Line with error value: " NR
		print "Difference: ", absd
		print "Tolerance: ", tol
		exit 1
	}else{
		print "OK line : ", NR, "Difference: ", diff
	}
}
'
