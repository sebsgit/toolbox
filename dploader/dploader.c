#include "dploader.h"

#ifdef _WIN32
    #include <windows.h>
    typedef HMODULE dplibhandle_priv_t_;
#elif defined __linux__
    #include <dlfcn.h>
    typedef void * dplibhandle_priv_t_;
#else
    #error Unsupported platform !
#endif

static void * load_priv_(const char * path){
#ifdef _WIN32
    return LoadLibraryA(path);
#elif defined __linux__
    dlerror();
    return dlopen(path,RTLD_LAZY | RTLD_LOCAL);
#endif
}

static void * get_sym_priv_(dplibhandle_priv_t_ h, const char * sym){
#ifdef _WIN32
    return GetProcAddress(h,sym);
#elif defined __linux__
    dlerror();
    return dlsym(h,sym);
#endif
}

static const char * get_err_priv_() {
    #ifdef _WIN32
        static char buff[256];
        snprintf(buff,256,"ERROR CODE: %i\n",GetLastError());
        return buff;
    #elif defined __linux__
        return dlerror();
    #endif
}

static void close_priv_(dplibhandle_priv_t_ h){
    if (h){
    #ifdef _WIN32
        FreeLibrary(h);
    #elif defined __linux__
        dlclose(h);
    #endif
    }
}


void * dp_load(const char * path){
    return load_priv_(path);
}

void * dp_symbol(void * h, const char * sym){
    dplibhandle_priv_t_ p = (dplibhandle_priv_t_)h;
    return get_sym_priv_(p,sym);
}

const char * dp_error(){
    return get_err_priv_();
}

void dp_close(void * h){
    dplibhandle_priv_t_ p = (dplibhandle_priv_t_)h;
    close_priv_(p);
}
