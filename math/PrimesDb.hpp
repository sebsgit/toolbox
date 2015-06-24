#ifndef PRIMESDB_HPP
#define PRIMESDB_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include <cmath>
#include <set>
#include "math_utils.hpp"

class BitBuffer{
public:
	typedef unsigned long long lint_t;
	BitBuffer(lint_t n_bits, bool init_value=true){
		_buffer.push_back( std::vector<bool>() );
		const lint_t part_size = std::min<lint_t>(n_bits,_buffer[0].max_size());
		std::cout << "setup buffer for " << n_bits << " bits...\n";
		_buffer[0].resize(part_size,init_value);
		n_bits -= part_size;
		while (n_bits){
			const lint_t to_alloc = std::min<lint_t>(n_bits,_buffer[0].max_size());
			std::vector<bool> buff(to_alloc, init_value);
			n_bits -= to_alloc;
			_buffer.push_back(buff);
		}
		std::cout << "created " << _buffer.size() << " buffers...\n";
		if (_buffer.size() > 1){
			_part_size = part_size;
		} else {
			_part_size = 0;
		}
	}
	bool operator[] (const lint_t index) const{
		if (_part_size == 0) return _buffer[0][index];
		else return _buffer[index/_part_size][index%_part_size];
	}
	void setBit(const lint_t index, const bool bit){
		if (_part_size == 0) _buffer[0][index] = bit;
		else _buffer[index/_part_size][index%_part_size] = bit;
	}
private:
	std::vector< std::vector<bool> > _buffer;
	lint_t _part_size;
};

template <typename Fun = std::function<void(unsigned long long)>>
class PrimeSieveGenerator{
	typedef unsigned long long lint_t;
public:
	PrimeSieveGenerator(Fun callback, const lint_t limit=75000000){
		const lint_t limit_half = limit/2;
		callback(2);
		BitBuffer tmp(limit_half,true);
		std::cout << "generating sieve...\n";
		for (lint_t i=0 ; i<limit_half ; ++i){
			if (tmp[i]){
				callback(2*i+3);
				lint_t m = (2*i+3)*3;
				while (m < limit){
					tmp.setBit( (m-3)/2, false );
					m += 2*(2*i+3);
				}
			}
		}
	}
};

template <typename Num=unsigned>
class PrimesDb{
	typedef Num lint_t;
public:
	PrimesDb(const lint_t limit = 75000000){
		auto f = [&](const lint_t p){_primes.push_back(p);};
		PrimeSieveGenerator<decltype(f)> generator(f,limit);
		std::cout << "generated " << _primes.size() << " primes.\n";
		std::cout << _primes[0] << " - " << _primes[_primes.size()-1] << "\n";
	}
	
	bool is_prime(const lint_t number) const{
		if (number > _primes[_primes.size()-1]){
			std::cout << "out of primes: " << number << ", max: " << _primes[_primes.size()-1] << "\n";
			throw 42;
		}
		return MUtils::contains(_primes,number);
	}
	
	bool is_prime_opt(lint_t number, int * index) const{
		if (number > _primes[_primes.size()-1]){
			std::cout << "out of primes: " << number << ", max: " << _primes[_primes.size()-1];
			throw 42;
		}
		for (size_t i=*index ; i<_primes.size() ; ++i){
			if (_primes[i] == number){
				*index = i;
				return true;
			} else if(_primes[i] > number){
				*index = i;
				return false;
			}
		}
		return false;
	}
	
	template <typename T, typename S>
	bool is_hamming(T n, const S type=5){
		for (size_t i=0 ; i<_primes.size() ; ++i){
			if (_primes[i] > n)
				return true;
			const bool is_ok =  n%_primes[i]==0;
			if (_primes[i] > type && is_ok){
				return false;
			} else if (is_ok){
				n /= _primes[i];
			}
		}
		return true;
	}
	
	template <typename T=int>
	T get_prime(unsigned n) const{
		return T(_primes.at(n));
	}
	
	std::vector<lint_t> primes() const{
		return _primes;
	}
	
	template <typename T>
	bool is_admissible(T n) const{
		if ((n&(n-1))==0){ // if power of 2
			return true;
		} else {
			for (size_t k=0 ; _primes[k]<n ; ++k){
				if (n>1 && n%_primes[k]!=0)
					return false;
				do{
					n /= _primes[k];
				} while (n%_primes[k]==0);
			}
			return true;
		}
		return false;
	}
	
	template <typename S, typename T>
	S first_prime_greater_than(const T t) const{
		const int i = MUtils::index_of_gt(_primes,t);
		return S( i>-1 ? _primes[i] : 0 );
	}
	
	template <typename T=int, typename R=int>
	T get_first_prime_below_eq(const R& r){
		for (size_t i=0;i<_primes.size();++i){
			if (_primes[i] == r) return _primes[i];
			else if (_primes[i] > r){
				if (i > 0){
					return T(_primes[i-1]);
				} else{
					return 0;
				}
			}
		}
		return T();
	}
	
	template <typename T=int, typename R=int>
	T get_first_prime_above_eq(const R& r){
		for (size_t i=0;i<_primes.size();++i){
			if (_primes[i] == r) return _primes[i];
			else if (_primes[i] > r){
				return T(_primes[i]);
			}
		}
		return T();
	}
	
	template <typename T=int, typename R=int>
	std::pair<T,T> get_clamp(const R& r){
		std::pair<T,T> result;
		for (size_t i=0;i<_primes.size();++i){
			if (_primes[i] == r){
				result.first = T(_primes[i]);
				result.second = T(_primes[i]);
				break;
			}
			else if (_primes[i] > r){
				result.first = T(_primes[i-1]);
				result.second = T(_primes[i]);
				break;
			}
		}
		return result;
	}
	
	template <typename T=int, typename R=int>
	std::pair<T,T> get_clamp_opt(const R& r, size_t * index){
		std::pair<T,T> result;
		for (size_t i=*index;i<_primes.size();++i){
			if (_primes[i] == r){
				result.first = T(_primes[i]);
				result.second = T(_primes[i]);
				*index = i;
				break;
			}
			else if (_primes[i] > r){
				result.first = T(_primes[i-1]);
				result.second = T(_primes[i]);
				*index = i;
				break;
			}
		}
		return result;
	}
	
	template <typename T=int>
	std::vector<T> get_primes_below(unsigned x) const{
		std::vector<T> result;
		for (size_t i=0;i<_primes.size();++i){
			if (_primes[i] >= x)
				break;
			result.push_back(T(_primes[i]));
		}
		return result;
	}
	
	template <typename T> 
	std::vector<T> convert_primes_below(unsigned x) const{
		std::vector<T> result;
		for (size_t i=0;i<_primes.size();++i){
			if (_primes[i] >= x)
				break;
			result.push_back(T(_primes[i]));
		}
		return result;
	}
	
	template <typename T>
	std::vector<T> convert_n_digit_primes(const size_t n_digits) const{
		std::vector<T> result;
		const long low_limit = pow(10L,n_digits-1);
		const long high_limit = pow(10L,n_digits);
		for (size_t i=0 ; i<_primes.size() ; ++i){
			if (_primes[i] > low_limit && _primes[i] < high_limit){
				result.push_back(T(_primes[i]));
			}
			if (_primes[i] > high_limit)
				break;
		}
		return result;
	}
	
	int phi(const unsigned x) const{
		//std::cout << x << " " << (x%2) << "\n";
        long double result = x;
		for (int i=0 ; _primes[i]<=x; ++i){
			//std::cout << "check: " << _primes[i] << ", " << (x%_primes[i]) << "\n";
			if ((x%_primes[i])==0){
                result = result*((_primes[i]-1.0L)/_primes[i]);
			}
		}
		return (int)result;
	}
	
	int num_primes_below_eq(unsigned x){
		int result = 0;
		for (int i=0 ; _primes[i]<=x; ++i){
			++result;
		}
		return result;
	}
	size_t num_primes() const{
		return _primes.size();
	}
	
	template <typename T>
	std::vector< std::pair<int,int> > factorize(T x) const{
		std::vector< std::pair<int,int> > result;
		for (int i=0 ; (T)_primes[i]<=x ; ++i){
			if (x%(T)_primes[i]==0){
				x /= (T)_primes[i];
				std::pair<int,int> cnt = std::make_pair(_primes[i],1);
				while (x%(T)_primes[i]==0){
					cnt.second += 1;
					x /= (T)_primes[i];
				}
				result.push_back(cnt);
			}
		}
		return result;
	}
	
	template <typename T>
	std::set<T> distinct_factors(T x) const{
		std::set< T > result;
		for (int i=0 ; (T)_primes[i]<=x ; ++i){
			if (x%(T)_primes[i]==0){
				x /= (T)_primes[i];
				result.insert(T(_primes[i]));
			}
		}
		return result;
	}
	
	template <typename T = long long, typename S = int>
	T radical(const S n) const{
		std::vector<std::pair<int,int> > fct = factorize(n);
		T total=1;
		for (size_t i=0 ; i<fct.size() ; ++i){
			total *= fct[i].first;
		}
		return total;
	}
	std::vector<lint_t> n_digit_primes(int n) const{
		std::vector<lint_t> result;
		int tmp;
		for (unsigned i=0 ; i<_primes.size() ; ++i){
			tmp = num_digits(_primes[i]);
			if (n == tmp)
				result.push_back(_primes[i]);
			else if (tmp > n)
				break;
		}
		return result;
	}
	static int num_digits(int x){
		int result = 1;
		while (x /= 10)
			++result;
		return result;
	}
	
private:
	std::vector<lint_t> _primes;
};

#endif
