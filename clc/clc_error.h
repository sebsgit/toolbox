#ifndef _CLC_ERROR_H_
#define _CLC_ERROR_H_

#ifdef __cplusplus
extern "C" {
#endif

#define CLC_ERR_UNKNOWN_ERR -1
#define CLC_ERR_NO_ERR 0
#define CLC_ERR_WRONG_PASS 1
#define CLC_ERR_FILE_OPEN_R 2
#define CLC_ERR_FILE_OPEN_W 3

extern void clc_set_err(int);
extern int clc_get_err();
extern char * clc_get_err_string();

#define _CLC_EXIT_OK_ clc_set_err(CLC_ERR_NO_ERR);

#ifdef __cplusplus
}
#endif

#endif
