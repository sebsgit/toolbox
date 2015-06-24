#include "clc_error.h"
#include <stdio.h>
#include <string.h>

static int _clc_err_code = CLC_ERR_NO_ERR;
static char _clc_err_string[256] = {0};

void clc_set_err(int err_code){
	_clc_err_code = err_code;
	switch(err_code){
	case CLC_ERR_UNKNOWN_ERR:
	sprintf(_clc_err_string,"CLC_ERR_UNKNOWN_ERR: unknown error");
	break;
	case CLC_ERR_NO_ERR:
	memset(_clc_err_string,0,256);
	break;
	case CLC_ERR_FILE_OPEN_R:
	sprintf(_clc_err_string,"CLC_ERR_FILE_OPEN_R: cannot open file for reading");
	break;
	case CLC_ERR_FILE_OPEN_W:
	sprintf(_clc_err_string,"CLC_ERR_FILE_OPEN_W: cannot open file for writing");
	break;
	case CLC_ERR_WRONG_PASS:
	sprintf(_clc_err_string,"CLC_ERR_WRONG_PASS: password do not match");
	break;
	}
}

int clc_get_err(){
	return _clc_err_code;
}

char * clc_get_err_string(){
	return _clc_err_string;
}
