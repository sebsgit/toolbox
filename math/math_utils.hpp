#ifndef INF_PREC_NUM_H_PP_
#define INF_PREC_NUM_H_PP_

#include <string>
#include <vector>
#include <set>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iterator>

namespace MUtils{
	
	template <typename T>
	T gcd(const T t1, const T t2){
		T c, a=t1, b=t2;
		while (b!=0){
			c = b;
			b = a%b;
			a = c;
		}
		return a;
	}
	
	// 1, -1, 2, -2,...
	// 1,  2, 5,  7, 12, 15, 22,...
	template <typename T=long long>
	T pentagonal(const T m){
		return (m*(3*m-1))/2;
	}
	
	// simplify a/(b*sq(N)+c) -> (h*sq(N)+v)/z
	template<typename T>
	void simplify(const T N, 
				  const T a, 
				  const T b, 
				  const T c, 
				  T * h, 
				  T * v,
				  T * z)
	{
		*h = a*b;
		*v = -a*c;
		*z = b*b*N-c*c;
	}
	
	template <typename T>
	void simplify(T * a, T * b){
		const T p = MUtils::gcd(*a,*b);
		if (p>1){
			*a /= p;
			*b /= p;
		}
	}

	template <typename T>
	void simplify(T * a, T * b, T * c){
		T p = MUtils::gcd(*a,*b);
		if ( p > 1){
			p = MUtils::gcd(*c,p);
			if (p>1){
				*a /= p;
				*b /= p;
				*c /= p;
			}
		}
	}
	
	template <typename T>
	bool has_period(const std::vector<T>& an, size_t len){
		for (size_t i=0 ; ; i+=len){
			if (i+len < an.size()){
				for (size_t a=0 ; a<len ; ++a){
					if (an[a] != an[i+a]){
						return false;
					}
				}
			} else{
				break;
			}
		}
		return true;
	}

	template <typename T>
	size_t find_period(const std::vector<T>& an){
		for (size_t sqLen=1 ; sqLen<an.size() ; ++sqLen)
			if (has_period(an,sqLen))
				return sqLen;
		return 0;
	}
	
	template <typename T>
	std::vector<T> continued_fractions(const T& n, size_t max_size=5000){
		std::vector<T> result;
		const T a0 = sqrt(n);
		T ai = a0;
		T a = 1;
		T b = 1;
		T c = -a0;
		T h,v,z;
		size_t i=0;
		result.push_back(a0);
		while (i++<max_size){
			simplify(n,a,b,c,&h,&v,&z);
			ai = (a0*h+v)/z;
			result.push_back(ai);
			a = z;
			b = h;
			c = v-ai*z;
			simplify(&a,&b,&c);
		}
		return result;
	}
	
	template <typename T, typename S>
	bool contains(const std::vector<T>& vec, const S& elem){
		return std::binary_search(vec.begin(), vec.end(), elem);
	}
	
	// finds index of element in SORTED vector via binary division
	template <typename T, typename S>
	int index_of(const std::vector<T>& vec, const S& elem){
		const auto it = std::lower_bound(vec.begin(), vec.end(), elem);
		if (it != vec.end()){
			return *it==elem ? it-vec.begin() : -1;
		}
		return -1;
	}
	
	// finds index of first greater than element in SORTED vector via binary division
	template <typename T, typename S>
	int index_of_gt(const std::vector<T>& vec, const S& elem){
		const auto it = std::upper_bound(vec.begin(), vec.end(), elem);
		if (it != vec.end()){
			return it-vec.begin();
		}
		return -1;
	}
	
	template<typename T>
	T digital_sum(T n){
		T sum=0;
		while (n){
			sum += n%10;
			n /= 10;
		}
		return sum;
	}
	// x!
	template <typename T>
	T factorial(T x){
		if (x<2) return 1;
		T result = 1;
		for (T i=2 ; i<=x ; ++i){
			result *= i;
		}
		return result;
	}
	
	template <typename T>
	T sum_digits_factorial(T x){
		if (x==0) return 1;
		T result=0;
		while (x){
			result += factorial(x%10);
			x /= 10;
		}
		return result;
	}
	
	// a^b % c
	template <typename T>
	T modulo(T a, T b, T c){
		T result = a%c;
		if (result == 0)
			return result;
		for (T i=1 ; i<b ; ++i){
			result = (result*a)%c;
		}
		return result;
	}
	
	template <typename T>
	T fast_mod(T a, T b, T c){
		if (b == 0){
			return 1;
		} else if (b == 1){
			return a%c;
		} else if (b%2 == 1){
			return (a*fast_mod<T>((a*a)%c,(b-1)/2,c))%c;
		} else {
			return fast_mod<T>((a*a)%c,b/2,c);
		}
	}
	
	template <typename T>
	bool can_be_triangle(const T a, const T b, const T c){
		return (a+b>c && a+c>b && b+c>a);
	}
	template <typename T>
	bool is_right_triangle(const T a, const T b, const T c){
		const T aa = a*a;
		const T bb = b*b;
		const T cc = c*c;
		return (aa+bb==cc || aa+cc==bb || bb+cc==aa);
	}
	
	// checks if given number is a perfect square
	// returns sqrt(x) if x is perfect square, 0 otherwise
	template <typename num_t>
	num_t perfect_square(const num_t x){
		if (x==1){
			return 1;
		}
		num_t test = x/2;
		num_t a = 0;
		num_t b = x;
		num_t tmp=0;
		while (1){
			const num_t res = test*test;
			if (res == x){
				return test;
			}else if (a==b || b < a || tmp==test){
				return 0;
			} else if (res > x){
				b = test;
				tmp = test;
				test = (a+b)/2;
			} else if (res < x){
				a = test;
				tmp = test;
				test = (a+b)/2;
			}
		}
	}
	
}

class Counter{
public:
	Counter(size_t n=1, int max=9, int min=0) : _max(max), _min(min){
		for (size_t i=0 ; i<n ; ++i)
			_data.push_back(_min);
	}
	bool at_end() const{
		for (size_t i=0 ; i<_data.size() ; ++i)
			if (_data[i] != _max)
				return false;
		return true;
	}
	std::vector<int> value() const{
		return _data;
	}
	void increment(){
		if (_data[0] != _max){
			++_data[0];
		} else {
			size_t i=0;
			while (_data[i] == _max){
				_data[i] = _min;
				++i;
			}
			if (i < _data.size()){
				_data[i]++;
			}
		}
	}
	bool contains_value(const int x) const{
		return std::find(_data.begin(), _data.end(), x) != _data.end();
	}
private:
	std::vector<int> _data;
	int _max;
	int _min;
};

class InfiniteNumber{
public:
    explicit InfiniteNumber(const std::string& number = "0"){
		for (int i=number.size()-1 ; i>=0 ; --i){
			_digits.push_back((int)(number[i]-'0'));
		}
    }
    explicit InfiniteNumber(const std::vector<int>& values){
		_digits = values;
	}
    template <typename I>
    InfiniteNumber(I t){
		if (t < 0){
			t = -t;
			this->_sign = -1;
		}
        while (t){
            _digits.push_back(t%10);
            t /= 10;
        }
    }

	std::string to_string() const{
		std::string result;
		if (_sign < 0)
			result.push_back('-');
		for (int i=_digits.size()-1 ; i>=0 ; --i)
			result.push_back((char)(_digits[i]+'0'));
		return result;
	}
	
	InfiniteNumber pow(unsigned p) const{
		InfiniteNumber result = *this;
		while (--p) result = result * *this;
		return result;
	}
	
	int num_digits() const{
		return _digits.size();
	}
	int digit(int index) const{
		return index < num_digits() ? _digits[index] : 0;
	}
	void set_digit(int index, int value){
		if (index < num_digits())
			_digits[index] = value;
	}
	int last_digit() const{
		return _digits[0];
	}
	void remove_front_digit(){
		_digits.erase(_digits.begin()+_digits.size()-1);
	}
	bool all_digits_leq(int value) const{
		for (unsigned int i=0 ; i<_digits.size() ; ++i){
			if (_digits[i] > value){
				return false;
			}
		}
		return true;
	}
	bool pandigital_back() const{
		if (_digits.size() >= 9){
			std::set<int> s;
			for (int i=0 ; i<9 ; ++i){
				if (_digits[i] != 0)
					s.insert(_digits[i]);
				else
					return false;
			}
			return s.size()==9;
		}
		return false;
	}
	bool pandigital_front() const{
		if (_digits.size() >= 9){
			std::set<int> s;
			const int size = _digits.size();
			for (int i=0 ; i<9 ; ++i){
				if (_digits[size-i-1] != 0)
					s.insert(_digits[size-i-1]);
				else
					return false;
			}
			return s.size()==9;
		}
		return false;
	}
	int sum_digits() const{
		int sum=0;
		for (unsigned i=0 ; i<_digits.size() ; ++i){
			sum += _digits[i];
		}
		return sum;
	}
	int digital_root() const{
		int result = sum_digits();
		while (result > 9){
			result = MUtils::digital_sum(result);
		}
		return result;
	}
	template<typename T> T convert() const{
		T t(0);
		T p10=1;
		for (T i=0 ; i<_digits.size() ; ++i){
			t += p10*_digits[i];
			p10 *= 10;
		}
		return t;
	}
	std::vector<int> digits() const{
		return _digits;
	}	
	InfiniteNumber& operator++(){
		*this = *this+1;
		return *this;
	}
	InfiniteNumber& operator += (const InfiniteNumber& n){
		*this = *this + n;
		return *this;
	}
private:
	std::vector<int> _digits;
	int _sign=1;
	friend InfiniteNumber operator + (const InfiniteNumber& n1, const InfiniteNumber& n2);
	friend InfiniteNumber operator - (const InfiniteNumber& n1, const InfiniteNumber& n2);
	friend InfiniteNumber operator * (const InfiniteNumber& n1, const InfiniteNumber& n2);
	friend bool operator < (const InfiniteNumber& n1, const InfiniteNumber& n2);
	friend bool operator > (const InfiniteNumber& n1, const InfiniteNumber& n2);
	friend bool operator == (const InfiniteNumber& n1, const InfiniteNumber& n2);
	friend bool operator != (const InfiniteNumber& n1, const InfiniteNumber& n2);
	friend std::ostream& operator << (std::ostream& out, const InfiniteNumber& n);
};

std::ostream& operator << (std::ostream& out, const InfiniteNumber& n){
	std::reverse_copy(n._digits.begin(), n._digits.end(), std::ostream_iterator<int>(out));
	return out;
}

InfiniteNumber operator+ (const InfiniteNumber& n1, const InfiniteNumber& n2){
	InfiniteNumber result;
	result._digits.clear();
	int carry=0;
	const int len = std::max(n1.num_digits(), n2.num_digits());
	for (int i=0 ; i<len ; ++i){
		int tmp = carry+n1.digit(i)+n2.digit(i);
		if (tmp > 9){
			carry = tmp/10;
			tmp = tmp%10;
		} else{
			carry = 0;
		}
		result._digits.push_back(tmp);
	}
	if (carry != 0){
		result._digits.push_back(carry);
	}
	return result;
}

InfiniteNumber operator - (const InfiniteNumber& n1, const InfiniteNumber& n2){
	if (n1 == n2)
		return InfiniteNumber(0);
	InfiniteNumber result;
	result._digits.clear();
	InfiniteNumber top = (n1 > n2 ? n1 : n2);
	const InfiniteNumber& bottom = (n1 > n2 ? n2 : n1);
	for (int i=0 ; i<bottom.num_digits() ; ++i){
		if (top.digit(i) >= bottom.digit(i)){
			result._digits.push_back(top.digit(i)-bottom.digit(i));
		} else {
			int k=i+1;
			while (top.digit(k) == 0){
				top.set_digit(k,9);
				++k;
			}
			top.set_digit(k,top.digit(k)-1);
			result._digits.push_back(10+top.digit(i)-bottom.digit(i));
		}
	}
	for (int i=bottom.num_digits() ; i<top.num_digits() ; ++i){
		result._digits.push_back(top.digit(i));
	}
	while (!result._digits.empty() && result.digit(result._digits.size()-1) == 0)
		result._digits.erase(result._digits.begin()+result._digits.size()-1);
	return result;
}

InfiniteNumber operator* (const InfiniteNumber& n1, const InfiniteNumber& n2){
	InfiniteNumber result;
	result._digits.clear();
	const InfiniteNumber& top = (n1.num_digits() >= n2.num_digits() ? n1 : n2);
	const InfiniteNumber& bottom = (n1.num_digits() >= n2.num_digits() ? n2 : n1);
	for (int i=0 ; i<bottom.num_digits() ; ++i){
		int carry=0;
		InfiniteNumber tmp;
		tmp._digits.clear();
		for (int k=0 ; k<top.num_digits() ; ++k){
			int x = bottom.digit(i)*top.digit(k)+carry;
			if (x > 9){
				carry = x/10;
				x = x%10;
			} else{
				carry = 0;
			}
			tmp._digits.push_back(x);
		}
		if (carry != 0){
			tmp._digits.push_back(carry);
		}
		tmp._digits.insert(tmp._digits.begin(),i,0);
		result = result+tmp;
	}
	return result;
}

bool operator < (const InfiniteNumber& n1, const InfiniteNumber& n2){
	if (n1.num_digits() < n2.num_digits()){
		return true;
	} else if (n1.num_digits() > n2.num_digits()){
		return false;
	} else{
		for (int i=n1.num_digits()-1 ; i>=0 ; --i){
			if (n1.digit(i) < n2.digit(i)){
				return true;
			} else if (n1.digit(i) > n2.digit(i)){
				return false;
			}
		}
		return false;
	}
}

bool operator > (const InfiniteNumber& n1, const InfiniteNumber& n2){
	return !(n1<n2) && !(n1==n2);
}

bool operator == (const InfiniteNumber& n1, const InfiniteNumber& n2){
	return n1._digits == n2._digits;
}

bool operator != (const InfiniteNumber& n1, const InfiniteNumber& n2){
	return !(n1==n2);
}

namespace MUtils{

template<typename T>
	bool is_palindrome(T i){
		const T input = i;
		T reverted = 0;
		while (i){
			reverted = reverted*10 + i%10;
			i /= 10;
		}
		return reverted==input;
	}

	typedef struct{
		InfiniteNumber a;
		InfiniteNumber b;
	} fract_t;

	static void flip(fract_t& f){
		std::swap(f.a,f.b);
	}

	static fract_t add_num(const fract_t& f, int x){
		fract_t result;
		result.b = f.b;
		result.a = f.a + f.b*InfiniteNumber(x);
		return result;
	}

	std::ostream& operator<< (std::ostream& out, const fract_t& f){
		out << f.a.to_string() << "/" << f.b.to_string();
		return out;
	}

	unsigned long long e(int n){
		return ((n-1)%3==0)? 2*(((n-1)/3)+1) : 1;
	}

	fract_t conv_e(int total_n){
		if (total_n==0){
			return fract_t{2,1};
		}
		--total_n;
		fract_t result{1,e(total_n)};
		--total_n;
		while (total_n >= 0){
			result = add_num(result, e(total_n));
			flip(result);
			--total_n;
		}
		result = add_num(result,2);
		return result;
	}

	template<typename T>
	bool pandigital_1_9(T x){
		if (x==0) return false;
		bool num_array[] = {true, false, false,
							false, false, false, 
							false, false,false,
							false};
		T tmp;
		while (x){
			tmp = x%10;
			if (num_array[tmp]) return false;
			num_array[tmp] = true;
			x /= 10;
		}
		for (int i=0 ; i<10 ; ++i) {
			if (!num_array[i]) 
				return false;
		}
		return true;
	}

	template<typename T>
	int num_digits(T x){
		if (x==0) return 1;
		int result=0;
		while (x){
			++result;
			x /= 10;
		}
		return result;
	}

	// find max x : x(20p+x) <= c
	template <typename T>
	T find_x(T p, T c){
		T tmp=0;
		for (int x=0 ; ; ++x){
			if (x*(20*p+x) > c){
				return tmp;
			}
			tmp = x;
		}
		return 0;
	}

	template <typename T>
	std::vector<T> separate_pairs(T n){
		std::vector<T> result;
		while (n){
			result.push_back(n%100);
			n /= 100;
		}
		std::reverse(result.begin(),result.end());
		return result;
	}

	template <typename T>
	std::vector<int> expand_root(const T n, size_t precision=100){
		std::vector<int> result;
		std::vector<int> pairs = separate_pairs(n);
		size_t i=0;
		InfiniteNumber p=0;
		InfiniteNumber r=0;
		
		int k=precision+1;
		while (--k){
			const InfiniteNumber c = 100*r+(i <= pairs.size()-1 ? pairs[i] : 0);
			const int x = find_x(p,c).convert<int>();
			const InfiniteNumber y = x*(20*p+x);
			p = 10*p+x;
			r = c-y;
			result.push_back(x);
			if (r==0 && i==pairs.size()-1){
				break;
			}
			if (i < pairs.size()){
				++i;
			}
		}
		return result;
	}

}

#endif

