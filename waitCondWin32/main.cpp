//g++ main.cpp -fpermissive

#include <windows.h>
#include <process.h>
#include <iostream>
#include <cassert>
#include <vector>

typedef unsigned int (*func_ptr) (void *); 

class Thread {
public:
	Thread(func_ptr task) : _task(task){
	}
	bool start(){
		this->_backend = _beginthreadex(NULL, 0, this->_task, 0, 0, &this->_id);
		return this->_backend != INVALID_HANDLE_VALUE;
	}
	void wait(){
		WaitForSingleObject(this->_backend, INFINITE);
	}
private:
	func_ptr _task;
	HANDLE _backend;
	unsigned _id;
};

class MutexWin32 {
public:
	MutexWin32()
		:_backend(CreateMutex(NULL, false, NULL))
	{
	}
	~MutexWin32(){
		CloseHandle(this->_backend);
	}
	bool lock() {
		return WAIT_OBJECT_0 == WaitForSingleObject(this->_backend, INFINITE);
	}
	bool unlock(){
		ReleaseMutex(this->_backend);
		return true;
	}
private:
	HANDLE _backend;
};

class WaitConditionWin32 {
public:
	WaitConditionWin32(){
	}
	~WaitConditionWin32(){
		for (int i=0 ; i<this->_notifiedEvents.size() ; ++i)
			CloseHandle(this->_notifiedEvents[i]);
		for (int i=0 ; i<this->_waitingEvents.size() ; ++i)
			CloseHandle(this->_waitingEvents[i]);
	}
	bool wait(MutexWin32& mutex) {
		HANDLE event = CreateEvent(NULL, false, false, NULL);
		this->_mutex.lock();
		this->_waitingEvents.push_back(event);
		this->_mutex.unlock();
		mutex.unlock();
		return WaitForSingleObject(event, INFINITE) == WAIT_OBJECT_0;
	}
	void signal() {
		HANDLE toNotify = 0;
		this->_mutex.lock();
		if (this->_waitingEvents.empty()==false){
			toNotify = this->_waitingEvents[0];
			this->_waitingEvents.erase(this->_waitingEvents.begin());
		}
		this->_mutex.unlock();
		if (toNotify) {
			SetEvent(toNotify);
			this->_notifiedEvents.push_back(toNotify);
		}
	}	
private:
	MutexWin32 _mutex;
	std::vector<HANDLE> _waitingEvents;
	std::vector<HANDLE> _notifiedEvents;
};

MutexWin32 * mutex;
WaitConditionWin32 * waitCondition;

unsigned int result = 0;

unsigned thread_task(void *){
	int i=0;
	while (i++ < 1000) {
		mutex->lock();
		std::cout << "1111 inside task " << i << "\n";
		++result;
		mutex->unlock();
	}
}

unsigned thread_task2(void *){
	int i=0;
	while (i++ < 1000) {
		mutex->lock();
		std::cout << "2222 inside task 2 " << i << "\n";
		++result;
		mutex->unlock();
	}
}

unsigned thread_task_with_wait(void *) {
	mutex->lock();
	std::cout << "thread task entered...\n";
	waitCondition->signal();
	mutex->unlock();
	while (result) {
		mutex->lock();
		--result;
		mutex->unlock();
		SwitchToThread();
	}
}

int main(){
	mutex = new MutexWin32;
	waitCondition = new WaitConditionWin32;
	Thread thread(thread_task);
	Thread thread2(thread_task2);
	assert(thread.start());
	assert(thread2.start());
	thread.wait();
	thread2.wait();
	assert(result == 2000);
	
	std::cout << "with wait...\n";
	Thread thread3(thread_task_with_wait);
	mutex->lock();
	assert(thread3.start());
	waitCondition->wait(*mutex);
	thread3.wait();
	std::cout << "EXIT main " << result << "\n";
	return 0;
}
