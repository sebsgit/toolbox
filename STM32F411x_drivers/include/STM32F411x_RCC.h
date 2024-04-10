#ifndef STM32F411X_RCC_H_
#define STM32F411X_RCC_H_

#include "STM32F411x_defs.h"

typedef enum
{
	ST_RCC_SYS_CLOCK_HSI = 0x0,
	ST_RCC_SYS_CLOCK_HSE = 0x1,
	ST_RCC_SYS_CLOCK_PLL = 0x2
} ST_RCC_system_clock_t;

extern void ST_RCC_set_system_clock(ST_RCC_system_clock_t clock_source);
extern uint8_t ST_RCC_system_clock_select_status();
extern uint32_t ST_RCC_get_system_clock_frequency();
extern uint32_t ST_RCC_get_apb1_prescaler();
extern uint32_t ST_RCC_get_ahb1_prescaler();

#endif // STM32F411X_RCC_H_
