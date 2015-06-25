*** Various programming tools ***  
  
dpstring - string and stringlist 'classes' written in C, I prefer this over plain char * in C projects  
		   short strings are stack-allocated and longer uses simple reallocation scheme (size = size x 2 up to 2048 bytes,   size = size + 2048 after that)  
		   no additional libraries required  
thpool	 - simple thread pool implementation, with support of multi-parameter functions  
		   dyncall library required to compile and run  
clc      - ciphering utilities written in C (AES, md5, sha-1)  
math     - C++ code written while solving problems on projecteuler.net (primes, fast modulo, big numbers etc.)  
inips	 - .ini file loader in C  
  
